# CancelVoice: Privacy-Preserving Voice Anonymization

CancelVoice is a research system for privacy-preserving voice authentication using adversarially trained generative models and diffusion-based privacy filters. The goal is to protect speaker identity during authentication without compromising the utility of the voice signal — enabling secure, anonymized voice-based access control.

This repository contains research code organized for iterative experimentation across model architectures, privacy filters, and evaluation protocols.

## Research Overview

CancelVoice addresses a core tension in voice authentication: systems need enough speaker-identifying information to verify identity, yet must not expose or store raw biometric voice data. We approach this through:

- **Adversarial privacy filters** — generative models trained to suppress identity-linked features while preserving authentication-relevant characteristics.
- **Diffusion-based anonymization** — diffusion models applied as a post-processing privacy layer to obfuscate speaker traits.
- **Authentication under anonymization** — evaluating whether anonymized representations remain discriminative enough for reliable verification.

## Repository Structure

```text
.
|-- anonymization_pipeline/   # Adversarial and diffusion-based privacy filters
|-- notebooks/                # Experiment inspection, metric visualization, ablations
|-- scripts/                  # Data preparation and model training entrypoints
|-- source_separation/        # Speaker isolation experiments prior to anonymization
|-- voice_blurring/           # Prototype privacy filter methods
`-- requirements.txt          # Python dependencies
```

## Quick Start

1. Create and activate a Python environment (Python 3.9+ recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Preparing Data

Use `scripts/prepare_sonyc_vox_mixes.py` to generate speaker mixtures from SONYC and VoxCeleb for training and evaluation.

Default dataset paths:
- SONYC annotations: `datasets/sonyc-v1-dataset/annotations.csv`
- SONYC audio: `datasets/sonyc-v1-dataset/`
- VoxCeleb audio: `datasets/voxceleb1-audio-wav-files-for-india-celebrity/`
- VoxCeleb metadata: `datasets/voxceleb1-audio-wav-files-for-india-celebrity/vox1_meta.csv`

Output is written to `--out-dir` along with a manifest CSV.

```bash
python scripts/prepare_sonyc_vox_mixes.py --mode train --out-dir data/mixes_train
python scripts/prepare_sonyc_vox_mixes.py --mode eval --out-dir data/mixes_eval --eval-snr high
python scripts/prepare_sonyc_vox_mixes.py --mode eval --out-dir data/mixes_eval --use-training-vox
```

## Training the Anonymization Model

The core model uses a U-Net architecture to learn a mapping from raw speaker audio to a privacy-filtered representation. Training expects a CSV manifest with:
- `mix_wav`: path to input mixture WAV
- `voice_wav`: path to the corresponding target speaker WAV

Basic training:
```bash
python scripts/train_unet.py --manifest data/pairs.csv --checkpoint-dir checkpoints/run1
```

Full options:
```bash
python scripts/train_unet.py \
  --manifest data/pairs.csv \
  --checkpoint-dir checkpoints/run1 \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-3 \
  --device auto \
  --num-workers 4 \
  --seed 0
```

Trained model saved to: `checkpoints/<run>/unet_voice_sep.pt`

## Datasets

- [VoxCeleb1 (India Celebrity Subset) — Kaggle](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
- [SONYC Urban Sound Dataset — Zenodo](https://zenodo.org/records/3692954)

## References

- Cohen-Hadria et al. (2019). *Voice Anonymization.* [PDF](https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf)

## Research Context

This work is part of a broader research agenda on privacy-preserving AI systems. Related interests include federated learning, differential privacy, and multimodal authentication. For questions or collaboration inquiries, feel free to open an issue or reach out directly.
