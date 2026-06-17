"""End-to-end CancelVoice anonymization pipeline.

Takes a raw speaker voice clip, runs it through the trained anonymization
model (or baseline methods if no checkpoint is available), and returns a
structured result containing the original waveform, the anonymized waveform,
and the speaker embedding cosine distance as a privacy metric.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import torch


# ---------------------------------------------------------------------------
# Resolve repo root so both voice_anonymization/ and scripts/ are importable
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root containing voice_anonymization/."""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "voice_anonymization").is_dir():
        return candidate
    raise RuntimeError(
        "Could not locate voice_anonymization/ directory. "
        "Make sure this file lives inside the CancelVoice repo."
    )


_repo_root = _find_repo_root()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from voice_anonymization import low_pass_blur, mfcc_inversion_blur


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnonymizationResult:
    """Structured output from the CancelVoice anonymization pipeline.

    Attributes
    ----------
    sr
        Sample rate of all waveforms in Hz.
    original_waveform
        Raw input voice signal before anonymization.
    anonymized_waveform
        Output waveform with speaker identity suppressed.
    cosine_distance
        Cosine distance between the original and anonymized speaker embeddings.
        Range [0, 2] — higher values indicate stronger identity suppression.
        None if speechbrain is not installed or embeddings could not be extracted.
    method
        Anonymization method used: 'cancelvoice_model' or 'blurring_baseline'.
    epoch
        Training epoch of the loaded checkpoint. None if baseline was used.
    """

    sr: int
    original_waveform: np.ndarray
    anonymized_waveform: np.ndarray
    cosine_distance: Optional[float] = field(default=None)
    method: str = field(default="unknown")
    epoch: Optional[int] = field(default=None)

    def privacy_summary(self) -> str:
        """Return a human-readable summary of the privacy metric."""
        lines = [
            "CancelVoice Anonymization Result",
            "-" * 34,
            f"Method          : {self.method}",
            f"Sample rate     : {self.sr} Hz",
            f"Duration        : {len(self.original_waveform) / self.sr:.2f}s",
        ]
        if self.epoch is not None:
            lines.append(f"Checkpoint epoch: {self.epoch}")
        if self.cosine_distance is not None:
            lines.append(f"Cosine distance : {self.cosine_distance:.4f}")
            if self.cosine_distance > 0.4:
                lines.append("Privacy level   : strong identity suppression")
            elif self.cosine_distance > 0.2:
                lines.append("Privacy level   : moderate suppression")
            else:
                lines.append("Privacy level   : weak suppression — model needs further training")
        else:
            lines.append("Cosine distance : unavailable (install speechbrain)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Speaker embedding extraction
# ---------------------------------------------------------------------------

def _extract_embedding(
    y: np.ndarray,
    sr: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    """Extract x-vector speaker embedding via SpeechBrain.

    Returns None if speechbrain is not installed or extraction fails.
    Install with: pip install speechbrain
    """
    try:
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="/tmp/cancelvoice_xvect",
            run_opts={"device": str(device)},
        )
        tensor = torch.tensor(y).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(tensor)
        return emb.squeeze().cpu().numpy()
    except ImportError:
        return None
    except Exception:
        return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. Higher = more dissimilar speakers."""
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Path, device: torch.device):
    """Load trained CancelVoiceModel from a checkpoint file."""
    from scripts.train_cancelvoice import CancelVoiceModel

    state      = torch.load(checkpoint_path, map_location=device)
    n_speakers = state.get("n_speakers", 1211)
    model      = CancelVoiceModel(n_speakers=n_speakers).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state.get("epoch")


def _model_anonymize(
    model,
    y: np.ndarray,
    sr: int,
    device: torch.device,
) -> np.ndarray:
    """Run the trained CancelVoice model on a single waveform."""
    mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_t    = torch.tensor(mel_norm.T, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        anon_mel, _, _, _ = model(mel_t)

    anon_mel       = anon_mel.squeeze(0).cpu().numpy().T
    anon_mel_power = librosa.db_to_power(anon_mel * mel_db.std() + mel_db.mean())
    # TODO: replace with trained HiFi-GAN vocoder for higher quality waveform output
    return librosa.feature.inverse.mel_to_audio(anon_mel_power, sr=sr, n_iter=64)


def _baseline_anonymize(y: np.ndarray, sr: int) -> np.ndarray:
    """Baseline anonymization: low-pass blurring followed by MFCC inversion."""
    y_lp, sr_lp = low_pass_blur(y, sr)
    return mfcc_inversion_blur(y_lp, sr_lp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def anonymize_audio(
    y: np.ndarray,
    sr: int,
    *,
    checkpoint: Optional[Path] = None,
    device: Optional[torch.device] = None,
    compute_embedding_distance: bool = True,
) -> AnonymizationResult:
    """Anonymize a voice waveform using CancelVoice.

    Runs the trained adversarial anonymization model if a checkpoint is
    provided and found on disk. Falls back to the baseline blurring methods
    (low-pass + MFCC inversion) from voice_anonymization/ otherwise.

    Parameters
    ----------
    y
        Input voice waveform as a float32 numpy array (mono, 16 kHz recommended).
    sr
        Sample rate of y in Hz.
    checkpoint
        Path to a trained CancelVoice checkpoint (.pt file).
        If None or the file does not exist, the baseline is used.
    device
        Torch device for model inference. Defaults to CUDA if available, else CPU.
    compute_embedding_distance
        Whether to compute the speaker embedding cosine distance metric.
        Requires speechbrain to be installed. Set to False to skip.

    Returns
    -------
    AnonymizationResult
        Dataclass containing the original waveform, anonymized waveform,
        cosine distance, method used, and checkpoint epoch if applicable.
    """
    y = np.asarray(y, dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose anonymization method
    use_model = checkpoint is not None and Path(checkpoint).is_file()
    epoch     = None

    if use_model:
        model, epoch = _load_model(Path(checkpoint), device)
        y_anon       = _model_anonymize(model, y, sr, device)
        method       = "cancelvoice_model"
    else:
        y_anon = _baseline_anonymize(y, sr)
        method = "blurring_baseline"

    # align length to original
    if len(y_anon) > len(y):
        y_anon = y_anon[: len(y)]
    elif len(y_anon) < len(y):
        y_anon = np.pad(y_anon, (0, len(y) - len(y_anon)))

    # compute speaker embedding cosine distance
    cosine_dist = None
    if compute_embedding_distance:
        emb_orig = _extract_embedding(y, sr, device)
        emb_anon = _extract_embedding(y_anon, sr, device)
        if emb_orig is not None and emb_anon is not None:
            cosine_dist = _cosine_distance(emb_orig, emb_anon)

    return AnonymizationResult(
        sr=sr,
        original_waveform=y,
        anonymized_waveform=y_anon,
        cosine_distance=cosine_dist,
        method=method,
        epoch=epoch,
    )
