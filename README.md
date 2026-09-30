# CancelVoice: Privacy-Preserving Voice Anonymization

CancelVoice is a research system for privacy-preserving voice anonymization using adversarially-trained generative models and diffusion-based privacy filters. The goal is to protect speaker identity during authentication without compromising the utility of the voice signal — enabling secure, anonymized voice-based access control.

This repository contains research code organized for iterative experimentation across model architectures, privacy filters, and evaluation protocols. The system is benchmarked against the VoicePrivacy 2024 Challenge standard.

## Research Overview

CancelVoice addresses a core tension in voice authentication: systems need enough speaker-identifying information to verify identity, yet must not expose or store raw biometric voice data. We approach this through:

- **Adversarial privacy filters** — generative models trained to suppress identity-linked features while preserving authentication-relevant characteristics.
- **Diffusion-based anonymization** — diffusion models applied as a post-processing privacy layer to obfuscate speaker traits.
- **Authentication under anonymization** — evaluating whether anonymized representations remain discriminative enough for reliable verification.

**Privacy guarantees targeted:**

- Voice unlinkability — two anonymized clips from the same speaker cannot be linked back to each other.
- Inversion attack resistance — the original speaker identity cannot be recovered from the anonymized output.
- Authentication utility — the anonymized voice retains sufficient features for legitimate verification.

## Repository Structure

```text
.
|-- anonymization_pipeline/   # Adversarial and diffusion-based privacy filters
|-- notebooks/                # Demo notebooks and experiment inspection
|-- scripts/                  # Data preparation, training, evaluation, and inference
|-- voice_anonymization/      # Baseline anonymization methods (low-pass and MFCC inversion)
`-- requirements.txt          # Python dependencies
```

## Quick Start

1. Create and activate a Python environment (Python 3.9+ recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Preparing Data

Use `scripts/prepare_data.py` to prepare the LibriSpeech evaluation set following the VoicePrivacy 2024 Challenge protocol. The script resamples clips to 16 kHz, filters by duration, and generates train/val/test manifest CSVs.

Default dataset path:
- LibriSpeech audio root: `datasets/librispeech/`

Output is written to `--out-dir` as three manifest CSVs.

```bash
python scripts/prepare_data.py \
  --data-root datasets/librispeech \
  --out-dir   data/prepared \
  --split     0.8 0.1 0.1
```

## Training the Anonymization Model

The CancelVoice model uses an adversarially-trained architecture that disentangles speaker identity features from linguistic content, suppresses the identity features via a privacy filter, and reconstructs the anonymized speech through a voice decoder.

Training expects the manifest CSVs produced by `prepare_data.py`.

Basic training:
```bash
python scripts/train_cancelvoice.py \
  --train-csv data/prepared/train.csv \
  --val-csv   data/prepared/val.csv \
  --checkpoint-dir checkpoints
```

Full options:
```bash
python scripts/train_cancelvoice.py \
  --train-csv      data/prepared/train.csv \
  --val-csv        data/prepared/val.csv \
  --checkpoint-dir checkpoints \
  --epochs         50 \
  --batch-size     32 \
  --lr             1e-4 \
  --lambda-adv     0.1 \
  --num-workers    4 \
  --seed           42
```

Best checkpoint saved to: `checkpoints/cancelvoice.pt`

## Anonymizing a Voice Clip

To anonymize a single audio file using a trained checkpoint:

```bash
python scripts/anonymize.py \
  --input      notebooks/demo.mp3 \
  --output     outputs/demo_anonymized.wav \
  --checkpoint checkpoints/cancelvoice.pt
```

If no checkpoint is available yet, the script falls back to the baseline anonymization methods in `voice_anonymization/` automatically.

## Evaluation

CancelVoice is evaluated using the VoicePrivacy 2024 Challenge protocol on the LibriSpeech test-clean split. The protocol defines three core metrics:

- **EER (Equal Error Rate)** — primary privacy metric. Measures whether a speaker verification system can distinguish speakers from anonymized speech. EER of 50% indicates complete anonymization (random-chance discrimination).
- **WER (Word Error Rate)** — linguistic utility metric. Measures transcription accuracy on anonymized speech via ASR. Lower is better.
- **UAR (Unweighted Average Recall)** — emotion preservation metric via Speech Emotion Recognition on IEMOCAP.

Run evaluation:
```bash
python scripts/evaluate.py \
  --test-csv   data/prepared/test.csv \
  --checkpoint checkpoints/cancelvoice.pt \
  --out-dir    results
```

## Benchmark

Evaluation follows the VoicePrivacy 2024 Challenge standard. The challenge evaluates systems on the LibriSpeech dev-clean and test-clean splits using an ECAPA-TDNN speaker verification attacker trained on LibriSpeech-train-960.

Target performance:
- EER > 40% (6 out of 36 VoicePrivacy 2024 submissions achieved this threshold)
- WER increase < 5% relative to original speech

## Datasets

- [LibriSpeech — OpenSLR](http://www.openslr.org/12) — used for privacy (EER) and utility (WER) evaluation following the VoicePrivacy 2024 protocol
- [IEMOCAP — USC SAIL Lab](https://sail.usc.edu/iemocap/) — used for emotion preservation (UAR) evaluation

## References

- Tomashenko et al. (2024). *The VoicePrivacy 2024 Challenge Evaluation Plan.* [PDF](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Eval_Plan_v1.0.pdf)
- Cohen-Hadria et al. (2019). *Voice Anonymization.* [PDF](https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf)

## Research Context

This work is part of a broader research agenda on privacy-preserving AI systems. Related interests include federated learning, differential privacy, and multimodal authentication. For questions or collaboration inquiries, feel free to open an issue or reach out directly.
