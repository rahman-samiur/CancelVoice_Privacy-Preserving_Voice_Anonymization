"""MFCC-inversion anonymization baseline (Cohen–Hadria et al., §2.2.2)."""

from __future__ import annotations

import numpy as np
import librosa
from librosa.feature.inverse import mfcc_to_audio


def mfcc_inversion_blur(
    y: np.ndarray,
    sr: float,
    *,
    n_mfcc: int = 5,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 32,
    dct_type: int = 2,
    norm: str | None = "ortho",
) -> np.ndarray:
    """Reconstruct audio from a truncated MFCC representation (approximate inversion).

    As in Cohen–Hadria et al. (2019), §2.2.2: keep only the first ``n_mfcc`` MFCCs
    (default 5), then invert via inverse DCT / mel mapping and Griffin–Lim as
    implemented in ``librosa.feature.inverse.mfcc_to_audio``.

    The STFT parameters below must match between ``librosa.feature.mfcc`` and
    ``mfcc_to_audio``; they are exposed so callers can stay consistent with a
    wider pipeline. Output length may differ slightly from ``len(y)``; trim or
    pad if sample alignment matters (e.g. remixing).

    Parameters
    ----------
    y
        Time-domain signal (mono or stereo; stereo is mixed to mono).
    sr
        Sample rate in Hz.
    n_mfcc
        Number of MFCC coefficients to retain (paper uses 5).
    n_mels
        Number of mel bands (forward and inverse).
    n_fft
        FFT length for the underlying STFT.
    hop_length
        STFT hop length in samples.
    n_iter
        Griffin–Lim iterations in ``mel_to_audio``.
    dct_type, norm
        Passed to both MFCC analysis and inversion (librosa defaults: 2 and
        ``'ortho'``).

    Returns
    -------
    np.ndarray
        Reconstructed waveform at ``sr``.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        dct_type=dct_type,
        norm=norm,
    )

    return mfcc_to_audio(
        mfcc,
        n_mels=n_mels,
        dct_type=dct_type,
        norm=norm,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
    )
