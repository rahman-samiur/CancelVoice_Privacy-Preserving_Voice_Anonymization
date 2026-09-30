# CancelVoice: Privacy-Preserving Voice Anonymization

CancelVoice is a research system for privacy-preserving voice anonymization using adversarially-trained generative models and diffusion-based privacy filters. The goal is to protect speaker identity during authentication without compromising the utility of the voice signal, enabling secure, anonymized voice-based access control.

This repository contains research code organized for iterative experimentation across model architectures, privacy filters, and evaluation protocols. The system is benchmarked against the VoicePrivacy 2024 Challenge standard.

## Research Overview

CancelVoice addresses a core tension in voice authentication: systems need enough speaker-identifying information to verify identity, yet must not expose or store raw biometric voice data. We approach this through:

- **Adversarial privacy filters:** generative models trained to suppress identity-linked features while preserving authentication-relevant characteristics.
- **Diffusion-based anonymization:** diffusion models applied as a post-processing privacy layer to obfuscate speaker traits.
- **Authentication under anonymization:** evaluating whether anonymized representations remain discriminative enough for reliable verification.

**Privacy guarantees targeted:**

- Voice unlinkability: two anonymized clips from the same speaker cannot be linked back to each other.
- Inversion attack resistance: the original speaker identity cannot be recovered from the anonymized output.
- Authentication utility: the anonymized voice retains sufficient features for legitimate verification.

## Adversary Model

CancelVoice is designed against four classes of adversary, each with a distinct capability and attack surface:

**1. Database adversary:** has access to a corpus of raw or anonymized voice recordings and attempts to re-identify speakers by matching embeddings across sessions. Mitigated by suppressing speaker-linked features in the identity branch so that embedding distances between clips from the same speaker approach those of different speakers (target EER ≥ 40%).

**2. Model-inversion adversary:** has white-box or black-box access to the anonymization model and attempts to reconstruct the original speaker embedding from the anonymized output. Mitigated by the privacy filter's non-invertible suppression of the identity subspace; the suppressed representation does not retain sufficient information to recover the original x-vector.

**3. Enrollment adversary:** has access to one or more enrollment utterances from a target speaker and attempts to verify whether a new anonymized clip belongs to the same speaker. Corresponds directly to the ECAPA-TDNN speaker verification attacker used in the VoicePrivacy 2024 evaluation protocol. Mitigated by maximizing confusion in the adversarial classifier (driving speaker logits toward a uniform distribution).

**4. Membership-inference adversary:** attempts to determine whether a specific speaker was present in the training data, using the model's outputs or internal representations as a side-channel. Mitigated by the adversarial training objective, which discourages the model from encoding speaker-specific patterns in its learned representations.

## Differential Privacy

CancelVoice provides a formal (ε, δ)-differential privacy bound on each anonymized output. The guarantee covers the person submitting a voice clip at inference time: for any two input clips that differ only in speaker identity, the anonymized outputs are statistically indistinguishable up to a multiplicative factor exp(ε) with failure probability δ.

The bound is realized through two components applied once per clip at inference:

- **Gaussian mechanism on the identity subspace:** calibrated noise with scale σ is injected into the L2-normalized identity embedding before decoding. The noise scale is chosen to satisfy (ε, δ)-DP under the Gaussian mechanism for a target ε ∈ {1, 5, 10}, with global sensitivity bounded by the L2 normalization step.
- **Single-application guarantee:** because noise is applied once per clip, the privacy cost is a single Rényi divergence value at the chosen noise scale, converted directly to the (ε, δ) guarantee via standard RDP-to-DP conversion (Mironov 2017). No composition over multiple steps is required.

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

- **EER (Equal Error Rate):** primary privacy metric. Measures whether a speaker verification system can distinguish speakers from anonymized speech. EER of 50% indicates complete anonymization (random-chance discrimination).
- **WER (Word Error Rate):** linguistic utility metric. Measures transcription accuracy on anonymized speech via ASR. Lower is better.
- **UAR (Unweighted Average Recall):** emotion preservation metric via Speech Emotion Recognition on IEMOCAP.

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

- [LibriSpeech — OpenSLR](http://www.openslr.org/12): used for privacy (EER) and utility (WER) evaluation following the VoicePrivacy 2024 protocol
- [IEMOCAP — USC SAIL Lab](https://sail.usc.edu/iemocap/): used for emotion preservation (UAR) evaluation

## References

- Tomashenko et al. (2024). *The VoicePrivacy 2024 Challenge Evaluation Plan.* [PDF](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Eval_Plan_v1.0.pdf)
- Cohen-Hadria et al. (2019). *Voice Anonymization.* [PDF](https://markcartwright.com/files/cohen-hadria2019voiceanonymization.pdf)
- Dwork & Roth (2014). *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in Theoretical Computer Science.
- Mironov (2017). *Rényi Differential Privacy of the Gaussian Mechanism.* IEEE CSF.

## Research Context

This work is part of a broader research agenda on privacy-preserving AI systems. Related interests include federated learning, differential privacy, and multimodal authentication. For questions or collaboration inquiries, feel free to open an issue or reach out directly.
