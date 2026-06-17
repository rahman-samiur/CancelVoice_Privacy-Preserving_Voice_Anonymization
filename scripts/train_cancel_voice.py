"""
train_cancelvoice.py

Training script for the CancelVoice adversarial anonymization model.

Loads speaker clips from a manifest CSV, trains the CancelVoiceModel
with a combined adversarial + utility loss, and saves checkpoints.

Usage:
    python scripts/train_cancelvoice.py \
        --train-csv  data/prepared/train.csv \
        --val-csv    data/prepared/val.csv \
        --checkpoint-dir checkpoints \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4

Checkpoints are saved to:
    checkpoints/cancelvoice_epoch{N}.pt
    checkpoints/cancelvoice.pt  (best validation loss)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
except ImportError:
    raise ImportError("Install librosa: pip install librosa")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VoiceDataset(Dataset):
    """
    Loads speaker clips from a manifest CSV and returns log-mel spectrograms.

    Each item is a dict with:
        mel       : (n_frames, n_mels) float32 tensor
        speaker_id: string label
    """

    def __init__(
        self,
        manifest_csv: Path,
        n_mels: int = 80,
        target_sr: int = 16000,
        fmax: int = 8000,
        max_frames: int = 256,
    ):
        self.n_mels     = n_mels
        self.target_sr  = target_sr
        self.fmax       = fmax
        self.max_frames = max_frames

        self.entries = []
        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append((row["speaker_id"], Path(row["clip_path"])))

        # build speaker-to-index mapping for adversarial classifier
        speakers = sorted(set(e[0] for e in self.entries))
        self.speaker_to_idx = {s: i for i, s in enumerate(speakers)}
        self.n_speakers = len(speakers)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        speaker_id, clip_path = self.entries[idx]

        y, _ = librosa.load(clip_path, sr=self.target_sr, mono=True)

        mel = librosa.feature.melspectrogram(
            y=y, sr=self.target_sr, n_mels=self.n_mels, fmax=self.fmax
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # normalise per clip
        mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        # transpose to (n_frames, n_mels) and pad/truncate to max_frames
        mel_norm = mel_norm.T
        if mel_norm.shape[0] > self.max_frames:
            mel_norm = mel_norm[: self.max_frames]
        else:
            pad = self.max_frames - mel_norm.shape[0]
            mel_norm = np.pad(mel_norm, ((0, pad), (0, 0)))

        return {
            "mel": torch.tensor(mel_norm, dtype=torch.float32),
            "speaker_idx": torch.tensor(self.speaker_to_idx[speaker_id], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FeatureEncoder(nn.Module):
    """Disentangles content features from speaker identity features."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.content_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.identity_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.content_branch(x), self.identity_branch(x)


class PrivacyFilter(nn.Module):
    """
    Suppresses speaker identity features adversarially.

    TODO: extend with diffusion-based stochastic perturbation
          in the identity subspace for stronger unlinkability.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.suppressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, identity_features):
        return self.suppressor(identity_features)


class VoiceDecoder(nn.Module):
    """Reconstructs anonymized mel spectrogram from content + suppressed identity."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 80):
        super().__init__()
        # TODO: replace with a neural vocoder (e.g. HiFi-GAN) for waveform output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, content, suppressed_identity):
        combined = torch.cat([content, suppressed_identity], dim=-1)
        return self.decoder(combined)


class SpeakerClassifier(nn.Module):
    """
    Auxiliary adversarial classifier.
    Trained to identify the speaker — the privacy filter is trained
    to fool this classifier (gradient reversal / minimax training).
    """

    def __init__(self, hidden_dim: int = 256, n_speakers: int = 1211):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_speakers),
        )

    def forward(self, identity_features):
        # pool across time dimension before classifying
        pooled = identity_features.mean(dim=1)
        return self.classifier(pooled)


class CancelVoiceModel(nn.Module):
    """Full CancelVoice anonymization pipeline."""

    def __init__(self, n_speakers: int = 1211):
        super().__init__()
        self.encoder         = FeatureEncoder()
        self.privacy_filter  = PrivacyFilter()
        self.decoder         = VoiceDecoder()
        self.adv_classifier  = SpeakerClassifier(n_speakers=n_speakers)

    def forward(self, mel):
        content, identity       = self.encoder(mel)
        suppressed_identity     = self.privacy_filter(identity)
        anonymized_mel          = self.decoder(content, suppressed_identity)
        speaker_logits          = self.adv_classifier(suppressed_identity)
        return anonymized_mel, speaker_logits, identity, suppressed_identity


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_loss(original_mel, anonymized_mel):
    """L1 loss on mel reconstruction — preserves linguistic content."""
    return nn.functional.l1_loss(anonymized_mel, original_mel)


def adversarial_loss(speaker_logits, speaker_idx, n_speakers):
    """
    Adversarial loss: penalise the model when the classifier correctly
    identifies the speaker from the suppressed identity features.
    We maximise classifier entropy (uniform distribution over speakers)
    as a proxy for unlinkability.
    """
    # uniform target distribution — the model should confuse the classifier
    uniform = torch.full_like(
        speaker_logits,
        fill_value=1.0 / n_speakers
    )
    log_probs = nn.functional.log_softmax(speaker_logits, dim=-1)
    return nn.functional.kl_div(log_probs, uniform, reduction="batchmean")


def classifier_loss(speaker_logits, speaker_idx):
    """Cross-entropy loss for the adversarial speaker classifier."""
    return nn.functional.cross_entropy(speaker_logits, speaker_idx)


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser_model, optimiser_clf, device, n_speakers, lambda_adv):
    model.train()
    total_recon = 0.0
    total_adv   = 0.0
    total_clf   = 0.0

    for batch in loader:
        mel         = batch["mel"].to(device)
        speaker_idx = batch["speaker_idx"].to(device)

        # forward pass
        anon_mel, speaker_logits, identity, suppressed = model(mel)

        # step 1: update speaker classifier to correctly identify speakers
        loss_clf = classifier_loss(speaker_logits.detach(), speaker_idx)
        optimiser_clf.zero_grad()
        # re-run classifier with fresh graph for its own backward pass
        _, logits_for_clf, _, suppressed_detached = model(mel)
        loss_clf = classifier_loss(logits_for_clf, speaker_idx)
        loss_clf.backward()
        optimiser_clf.step()

        # step 2: update anonymization model
        anon_mel, speaker_logits, _, _ = model(mel)
        loss_recon = reconstruction_loss(mel, anon_mel)
        loss_adv   = adversarial_loss(speaker_logits, speaker_idx, n_speakers)
        loss_total = loss_recon + lambda_adv * loss_adv

        optimiser_model.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser_model.step()

        total_recon += loss_recon.item()
        total_adv   += loss_adv.item()
        total_clf   += loss_clf.item()

    n = len(loader)
    return total_recon / n, total_adv / n, total_clf / n


def validate(model, loader, device, n_speakers, lambda_adv):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            mel         = batch["mel"].to(device)
            speaker_idx = batch["speaker_idx"].to(device)

            anon_mel, speaker_logits, _, _ = model(mel)
            loss_recon = reconstruction_loss(mel, anon_mel)
            loss_adv   = adversarial_loss(speaker_logits, speaker_idx, n_speakers)
            total_loss += (loss_recon + lambda_adv * loss_adv).item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train CancelVoice anonymization model.")
    parser.add_argument("--train-csv",      type=Path, required=True)
    parser.add_argument("--val-csv",        type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs",         type=int,  default=50)
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--lambda-adv",     type=float, default=0.1,
                        help="Weight for adversarial loss term (default: 0.1).")
    parser.add_argument("--num-workers",    type=int,  default=4)
    parser.add_argument("--seed",           type=int,  default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # datasets
    train_dataset = VoiceDataset(args.train_csv)
    val_dataset   = VoiceDataset(args.val_csv)
    n_speakers    = train_dataset.n_speakers

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    print(f"Train: {len(train_dataset)} clips | Val: {len(val_dataset)} clips")
    print(f"Speakers: {n_speakers}")

    # model
    model = CancelVoiceModel(n_speakers=n_speakers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # separate optimisers for anonymization model and adversarial classifier
    clf_params   = list(model.adv_classifier.parameters())
    model_params = [p for p in model.parameters() if not any(p is cp for cp in clf_params)]

    optimiser_model = optim.Adam(model_params, lr=args.lr)
    optimiser_clf   = optim.Adam(clf_params,   lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        recon, adv, clf = train_epoch(
            model, train_loader,
            optimiser_model, optimiser_clf,
            device, n_speakers, args.lambda_adv
        )
        val_loss = validate(model, val_loader, device, n_speakers, args.lambda_adv)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"recon={recon:.4f} adv={adv:.4f} clf={clf:.4f} | "
            f"val={val_loss:.4f} | {elapsed:.1f}s"
        )

        # save checkpoint every epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_model_state_dict": optimiser_model.state_dict(),
            "optimiser_clf_state_dict": optimiser_clf.state_dict(),
            "val_loss": val_loss,
            "n_speakers": n_speakers,
        }
        epoch_path = args.checkpoint_dir / f"cancelvoice_epoch{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = args.checkpoint_dir / "cancelvoice.pt"
            torch.save(checkpoint, best_path)
            print(f"  New best checkpoint saved: {best_path} (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
