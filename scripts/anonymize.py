"""
anonymize.py

Inference script for CancelVoice.

Takes a single audio file, runs it through the trained anonymization model,
and writes the anonymized output to disk.

Usage:
    python scripts/anonymize.py \
        --input      notebooks/demo.mp3 \
        --output     outputs/demo_anonymized.wav \
        --checkpoint checkpoints/cancelvoice.pt

If no checkpoint is available yet, the script falls back to the
blurring pipeline baseline (low-pass + MFCC inversion) so the
script remains useful during model development.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch


TARGET_SR = 16000


# ---------------------------------------------------------------------------
# Blurring baseline (used when no checkpoint is available)
# ---------------------------------------------------------------------------

def low_pass_blur(y: np.ndarray, sr: int, cutoff_hz: int = 500) -> tuple[np.ndarray, int]:
    """
    Downsample to cutoff_hz then resample back to sr.
    Removes high-frequency speaker identity cues.
    """
    y_down   = librosa.resample(y, orig_sr=sr, target_sr=cutoff_hz)
    y_up     = librosa.resample(y_down, orig_sr=cutoff_hz, target_sr=sr)
    return y_up, sr


def mfcc_inversion_blur(y: np.ndarray, sr: int, n_mfcc: int = 5) -> np.ndarray:
    """
    Extract first n_mfcc MFCCs then reconstruct via Griffin-Lim.
    Discards fine-grained spectral texture linked to speaker identity.
    """
    mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel_from_mfcc = librosa.feature.inverse.mfcc_to_mel(mfccs, n_mels=128)
    mel_from_mfcc = np.maximum(mel_from_mfcc, 0)
    y_out    = librosa.feature.inverse.mel_to_audio(mel_from_mfcc, sr=sr, n_iter=64)
    return y_out


def blurring_pipeline(y: np.ndarray, sr: int) -> np.ndarray:
    """Low-pass blurring followed by MFCC inversion — baseline anonymization."""
    y_lp, sr_lp = low_pass_blur(y, sr)
    y_anon      = mfcc_inversion_blur(y_lp, sr_lp)
    return y_anon


# ---------------------------------------------------------------------------
# Model-based anonymization
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained CancelVoiceModel from checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.train_cancelvoice import CancelVoiceModel

    state     = torch.load(checkpoint_path, map_location=device)
    n_speakers = state.get("n_speakers", 1211)
    model     = CancelVoiceModel(n_speakers=n_speakers).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state.get("epoch", "?")


def model_anonymize(model, y: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    """Run the trained CancelVoice model on a single waveform."""
    mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_t    = torch.tensor(mel_norm.T, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        anon_mel, _, _, _ = model(mel_t)

    anon_mel       = anon_mel.squeeze(0).cpu().numpy().T
    anon_mel_power = librosa.db_to_power(anon_mel * mel_db.std() + mel_db.mean())
    # TODO: replace with HiFi-GAN vocoder for higher quality waveform reconstruction
    y_anon         = librosa.feature.inverse.mel_to_audio(anon_mel_power, sr=sr, n_iter=64)
    return y_anon


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Anonymize a voice recording using CancelVoice."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input audio file (WAV or MP3).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for anonymized WAV. "
             "Defaults to <input_stem>_anonymized.wav in the same directory.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/cancelvoice.pt"),
        help="Path to trained model checkpoint. "
             "Falls back to blurring baseline if not found.",
    )
    parser.add_argument(
        "--method", choices=["model", "blur", "auto"], default="auto",
        help="Anonymization method: "
             "'model' uses the trained CancelVoice model, "
             "'blur' uses the blurring pipeline baseline, "
             "'auto' uses the model if available, else falls back to blur (default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    # default output path
    if args.output is None:
        args.output = args.input.parent / (args.input.stem + "_anonymized.wav")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # load input
    print(f"Loading: {args.input}")
    y, sr = librosa.load(args.input, sr=TARGET_SR, mono=True)
    print(f"Duration: {len(y) / sr:.2f}s | Sample rate: {sr} Hz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose anonymization method
    use_model = False
    if args.method == "model":
        use_model = True
    elif args.method == "auto":
        use_model = args.checkpoint.is_file()
        if not use_model:
            print(
                f"\nNo checkpoint found at {args.checkpoint}. "
                "Falling back to blurring pipeline baseline."
            )
            print("Train the model with: python scripts/train_cancelvoice.py\n")

    if use_model:
        print(f"\nLoading model checkpoint: {args.checkpoint}")
        model, epoch = load_model(args.checkpoint, device)
        print(f"Model loaded (trained for {epoch} epochs).")
        print("Running CancelVoice model inference...")
        y_anon = model_anonymize(model, y, sr, device)
        method_used = "cancelvoice_model"
    else:
        print("Running blurring pipeline baseline (low-pass + MFCC inversion)...")
        y_anon = blurring_pipeline(y, sr)
        method_used = "blurring_baseline"

    # write output
    # match length to original to avoid downstream issues
    if len(y_anon) > len(y):
        y_anon = y_anon[: len(y)]
    elif len(y_anon) < len(y):
        y_anon = np.pad(y_anon, (0, len(y) - len(y_anon)))

    sf.write(args.output, y_anon, sr, subtype="PCM_16")

    print(f"\nAnonymized output saved to: {args.output}")
    print(f"Method used: {method_used}")
    print(f"Duration: {len(y_anon) / sr:.2f}s")


if __name__ == "__main__":
    main()
