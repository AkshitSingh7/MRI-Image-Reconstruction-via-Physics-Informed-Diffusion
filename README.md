# MRI Image Reconstruction via Physics-Informed Diffusion

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **High-fidelity MRI reconstruction from 4x undersampled data using a Physics-Informed Conditional Diffusion Model. Implements a spectral Data Consistency (DC) layer during inference to enforce K-Space validity, achieving a +2.15 dB PSNR improvement.**

---

## 🧠 Project Overview
Magnetic Resonance Imaging (MRI) acquisition is slow, often leading to patient discomfort and motion artifacts. Accelerating scans by skipping frequencies (undersampling) introduces complex aliasing artifacts that standard CNNs often struggle to remove without blurring fine details.

This project implements a **Physics-Informed Diffusion Probabilistic Model (DDPM)** that treats reconstruction not just as image denoising, but as a constrained generation problem. By explicitly enforcing Fourier-space (K-Space) consistency, the model recovers high-frequency anatomical details that are mathematically consistent with the raw scanner data.

### 🔬 Key Results
We evaluated the model on the **IXI Brain Dataset** (T1-weighted) with 4x acceleration.

| Metric | Input (Blurry/Undersampled) | Physics-Guided AI Output | Improvement |
| :--- | :---: | :---: | :---: |
| **PSNR** (Peak Signal-to-Noise Ratio) | 28.01 dB | **30.16 dB** | **+2.15 dB** 🚀 |
| **SSIM** (Structural Similarity) | 0.82 | **0.91** | **+11%** |

---

## 🚀 The Process: Three-Layer Physics Integration
Unlike standard Generative AI which can "hallucinate" anatomical features, this pipeline bakes MRI physics into three distinct stages:

### 1. Training Simulation (The Forward Model)
Instead of adding generic Gaussian noise, the training data is generated via a differentiable **Spectral Undersampling Simulator**.
* **Method:** We apply a Fast Fourier Transform (FFT) to ground-truth images, mask 92% of the frequencies (simulating a 4x speedup), and transform back.
* **Result:** The model learns specifically to reverse aliasing and Gibbs ringing artifacts.

### 2. Conditioning (The Guide)
The model architecture is a **Conditional U-Net**.
* **Input:** `[Noisy_Latent, Blurry_Condition]` (2 Channels).
* **Mechanism:** The diffusion process is explicitly conditioned on the acquired (undersampled) image, acting as a structural guide for the generative process.

### 3. Inference Enforcement (Data Consistency)
We implement a hard **Data Consistency (DC) Projection Layer** during the reverse diffusion sampling.
* **Logic:** At every denoising step $t$, the predicted image is projected into K-Space.
* **Enforcement:** We replace the model's predicted frequencies with the *actual measured frequencies* from the scanner wherever they exist.
* **Benefit:** This guarantees the output mathematically matches the raw sensor data, preventing hallucinations.

---

## 📂 Directory Structure
```bash
MRI-Reconstruction-Diffusion/
├── checkpoints/             # Saved model weights (Epoch 50, 100)
├── dataset/                 # IXI Dataset (T1-weighted NIfTI files)
├── results/                 # Output images and metric logs
├── src/
│   ├── physics.py           # FFT simulator & Data Consistency functions
│   ├── model.py             # MONAI DiffusionModelUNet definition
│   └── utils.py             # PSNR/SSIM metrics and visualization
├── train.py                 # Main training loop (with Mixed Precision)
├── inference.py             # Physics-guided refinement script
├── requirements.txt         # Dependencies
└── README.md                # This file

```

---

## 💻 Tech Stack & Models

* **Framework:** PyTorch, MONAI (Medical Open Network for AI)
* **Model Architecture:**
* **Type:** Denoising Diffusion Probabilistic Model (DDPM)
* **Backbone:** U-Net with Attention Gates at resolutions 16/8
* **Channels:** `[64, 128, 256]` with `num_head_channels=32`


* **Compute:** Optimized for NVIDIA A100 GPUs using Automatic Mixed Precision (`fp16`).
* **Optimization:** Adam Optimizer, Cosine Annealing, Exponential Moving Average (EMA).

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/AkshitSingh7/MRI-Image-Reconstruction-via-Physics-Informed-Diffusion.git](https://github.com/AkshitSingh7/MRI-Image-Reconstruction-via-Physics-Informed-Diffusion.git)
cd MRI-Image-Reconstruction-via-Physics-Informed-Diffusion

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run Training

The script automatically downloads the IXI dataset and starts training.

```bash
python train.py --epochs 100 --batch_size 16

```

### 4. Run Physics-Guided Inference

To reconstruct a sample image using the Data Consistency strategy:

```bash
python inference.py --checkpoint checkpoints/model_epoch_100.pth --strategy refinement

```

---

## 📸 Visualization

### Comparison: Input vs. AI Reconstruction

*(See `results/` folder for full resolution)*

| **Input (4x Accel)** | **Physics-Guided AI** | **Ground Truth** | **Error Map** |
| --- | --- | --- | --- |
|  |  |  |  |

> *Note the restoration of the cerebellum folds and sharp ventricular boundaries in the AI output (Middle), which are blurred in the Input (Left).*

---

## 📜 Dataset & Citation

This project uses the **IXI Dataset** (Information eXtraction from Images), collected by:

* *Imperial College London*
* *Guy's Hospital*
* *Hammersmith Hospital*

Dataset Source: [https://brain-development.org/ixi-dataset/](https://brain-development.org/ixi-dataset/)

If you use this code, please credit the repository:

```
@misc{singh2024mridiffusion,
  author = {Akshit Singh},
  title = {Physics-Informed MRI Reconstruction via Diffusion Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/AkshitSingh7/MRI-Image-Reconstruction-via-Physics-Informed-Diffusion](https://github.com/AkshitSingh7/MRI-Image-Reconstruction-via-Physics-Informed-Diffusion)}}
}

```

```

```
