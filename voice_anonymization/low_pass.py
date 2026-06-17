"""Low-pass voice blurring via sample-rate conversion (Cohen–Hadria et al., §2.2.1)."""

from __future__ import annotations

import numpy as np
import librosa


def low_pass_blur(
    y: np.ndarray,
    sr: float,
    *,
    down_sr: float = 500.0,
    out_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Blur speech by band-limiting through aggressive downsampling and upsampling.

    As in Cohen-Hadria et al. (2019), §2.2.1: sub-sample to ``down_sr`` (500 Hz,
    i.e. ~250 Hz low-pass) then resample to the output rate (16 kHz in the paper)
    using librosa's sample-rate conversion.

    Parameters
    ----------
    y
        Time-domain signal (mono or stereo; stereo is mixed to mono).
    sr
        Sample rate of ``y`` in Hz.
    down_sr
        Intermediate sample rate after the first resample (default 500 Hz).
    out_sr
        Final sample rate (default 16000 Hz).

    Returns
    -------
    y_blur
        Blurred waveform at ``out_sr``.
    out_sr
        Same as the ``out_sr`` argument (returned for call-site symmetry).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)

    y_down = librosa.resample(y, orig_sr=sr, target_sr=down_sr)
    y_out = librosa.resample(y_down, orig_sr=down_sr, target_sr=float(out_sr))
    return y_out, int(out_sr)
