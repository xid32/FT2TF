
# FT2TF: First-Person Statement Text-to-Talking Face Generation

FT2TF is a state-of-the-art framework designed for text-driven talking face generation. Unlike traditional audio-driven methods, FT2TF leverages only visual and textual inputs to generate dynamic and expressive talking face videos. This repository contains the official implementation of the FT2TF pipeline.

---

## Features
- **One-Stage End-to-End Framework:** Combines visual and textual modalities for seamless face generation without intermediate steps.
- **Text-Driven Approach:** Eliminates the need for audio or landmark data, significantly reducing resource requirements.
- **Multi-Scale Cross-Attention Module:** Ensures robust fusion of emotional and linguistic text features with visual inputs.
- **State-of-the-Art Performance:** Outperforms existing methods on LRS2 and LRS3 datasets across multiple metrics.

---

## Code Running Structure
```
FT2TF/
├── preprocessing.py          # Video preprocessing
├── gpt_preprocessing.py      # Textual data preparation
├── emobert_processing.py     # EmoBERT feature extraction
├── project/
│   ├── train_final.py        # Main training script
│   ├── args                 # Configuration file
```

---

## Preprocessing

### Video Preprocessing
Standardize video frames to 96x96 resolution and extract facial regions:

```bash
python preprocessing.py
```

### Text Data Preparation
Generate linguistic and emotional text embeddings using GPT-Neo and EmoBERT models:

```bash
python gpt_preprocessing.py
python emobert_processing.py
```

---

## Training

Train the FT2TF model using:

```bash
python project/train_final.py
```

Ensure configurations in `project/args` are properly set for datasets, hyperparameters, and model architecture.

---

## Evaluation

Evaluate the model on LRS2 and LRS3 datasets using:
- **Metrics:** PSNR, SSIM, LPIPS, FID, LipLMD, and CSIM.
- **Comparison:** Benchmarked against state-of-the-art methods.

---

## Configuration

Modify training and model parameters in:

```text
project/args
```

## Acknowledgments

This work is based on the paper *"First-Person Statement Text-to-Talking Face Generation"*. Please cite the paper if you use this code:

```
@article{diao2023ft2tf,
  title={Ft2tf: First-person statement text-to-talking face generation},
  author={Diao, Xingjian and Cheng, Ming and Barrios, Wayner and Jin, SouYoung},
  journal={arXiv preprint arXiv:2312.05430},
  year={2024}
}
```

For questions or issues, contact [repository maintainer].
