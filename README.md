

# 🖼️ Deep Learning-Based Image Deblurring: A Technical Guide

This document provides a comprehensive technical guide to deep learning-based image deblurring methods, covering foundational concepts and a review of state-of-the-art models like SRCNN, DeblurGAN, SRN-Deblur, MPRNet, and HINet.

## Table of Contents

1.  [Introduction – What Is Image Blur?](#1-introduction--what-is-image-blur)
    1.  [Mathematical Model of Blur](#11-mathematical-model-of-blur)
    2.  [Physical Causes of Blur](#12-physical-causes-of-blur)
2.  [Blur Types & Mathematical Models](#2-blur-types--mathematical-models)
3.  [Deep Learning–Based Deblurring Methods](#3-deep-learningbased-deblurring-methods)
    1.  [SRCNN-Based Basic CNN](#31-srcnn-based-basic-cnn)
    2.  [DeblurGAN](#32-deblurgan)
    3.  [SRN-Deblur (Scale-Recurrent Network)](#33-srn-deblur-scale-recurrent-network)
    4.  [Deblurring by Realistic Blurring (BGAN + DBGAN)](#34-deblurring-by-realistic-blurring-bgan--dbgan)
    5.  [MPRNet (Multi-Stage Progressive Restoration)](#35-mprnet-multi-stage-progressive-restoration)
    6.  [HINet (Half Instance Normalization Network)](#36-hinet-half-instance-normalization-network)
4.  [Performance Comparison Tables](#4-performance-comparison-tables)
5.  [Summary & Practical Recommendations](#5-summary--practical-recommendations)
6.  [References](#6-references)

---

## 1. Introduction – What Is Image Blur?

Image blur is a common degradation that reduces spatial detail, lowering both human-perceivable quality and algorithmic performance. Recovering a sharp image from a blurred observation is called **deblurring**. If the blur kernel is unknown, the process becomes **blind deblurring**, where both the kernel and the latent image must be estimated.

### 1.1 Mathematical Model of Blur

The general blur model is formulated as a convolution between the clean image, a blur kernel, and additive noise:

$$
B(x, y) = (I * k)(x, y) + n(x, y)
$$

Where:
*   $I$: The original clean (sharp) image.
*   $k$: The blur kernel, also known as the Point Spread Function (PSF).
*   $*$: The convolution operator.
*   $n$: Additive noise (e.g., Gaussian noise).

### 1.2 Physical Causes of Blur

1.  **Uniform Motion Blur**: Occurs when the camera moves at a constant velocity and direction during exposure. All pixels share the same PSF.
2.  **Non-Uniform Motion Blur**: Arises when different objects or parts of the scene move with varying velocities and/or directions. The PSF is spatially varying ($k_{x,y}$).
3.  **Defocus Blur**: Happens when the lens fails to focus objects onto the sensor plane. The PSF is often approximated by a circular disk.
4.  **Atmospheric/Turbulence Blur**: Caused by random distortions from air currents or heat, common in long-range imaging.

---

## 2. Blur Types & Mathematical Models

| Blur Type | PSF (Mathematical Model) | Description |
| :--- | :--- | :--- |
| **Uniform Motion Blur** | PSF is a line segment of length $L$ and orientation $\theta$. | Camera (or entire scene) moves at constant velocity; same PSF for all pixels. |
| **Non-Uniform Motion Blur** | $B(x,y) = \sum_{u,v} I(u,v)\,k_{x,y}(x-u,\,y-v) + n(x,y)$ | Objects move in different directions or speeds; each pixel/region has its own PSF $k_{x,y}$. |
| **Defocus Blur** | $$ k(x, y) = \begin{cases} \frac{1}{\pi R^2}, & x^2 + y^2 \le R^2 \\ 0, & \text{otherwise.} \end{cases} $$ | Lens focusing error results in out-of-focus regions; PSF is a circular disk of radius $R$. |
| **Atmospheric/Turbulence** | Typically Gaussian: $k(x,y) \propto e^{-\frac{x^2+y^2}{2\sigma^2}}$ plus speckle noise. | Optical distortions due to heat or air currents; common in long-range or astronomical imaging. |

*   **Blind Deblurring**: Kernel $k$ is unknown; the model must estimate both $\hat{k}$ and $\hat{I}$.
*   **Non-Blind Deblurring**: Kernel $k$ is known; restoration can be applied directly.

---

## 3. Deep Learning–Based Deblurring Methods

### 3.1 SRCNN-Based Basic CNN

#### Approach Summary

A lightweight 3-layer convolutional network is trained to map Gaussian-blurred inputs back to sharp outputs. This architecture follows the SRCNN paradigm, using zero-padding to preserve spatial dimensions and MSE loss for pixel-wise reconstruction.

#### Architecture Details
*   **Input:** $224 \times 224$ blurred image (RGB).
*   **Layers:**
    1.  `Conv1`: 64 filters, $9 \times 9$ kernel, ReLU.
    2.  `Conv2`: 32 filters, $1 \times 1$ kernel, ReLU.
    3.  `Conv3`: 3 filters, $5 \times 5$ kernel, Linear output.
*   **Loss Function:** Mean Squared Error (MSE).
*   **Optimizer:** Adam ($lr = 10^{-3}$).

#### Training Setup
*   **Dataset:** Kaggle Blur Dataset (1,050 images).
*   **Data Augmentation:** Flips, crops to $224 \times 224$.
*   **Batch Size:** 16
*   **Epochs:** 40
*   **Hardware:** Single NVIDIA GTX 1080 Ti (≈ 4 hours).

#### Performance
*   **PSNR:** 27.5 dB
*   **SSIM:** 0.85
*   **Parameters:** ~60 K
*   **Inference Time:** ~0.012 s/image (on GPU)

> **Note:** This simple model is suitable for fast prototyping but struggles with complex, real-world blur.

### 3.2 DeblurGAN

#### Approach Summary
DeblurGAN employs a conditional Generative Adversarial Network (GAN) to transform a blurred image into a sharp one. The generator uses ResNet blocks and minimizes a combination of perceptual loss and adversarial loss, producing visually realistic results.

#### Architecture Details
*   **Generator:** A ResNet-based architecture with 2 downsampling blocks, 9 ResNet blocks, and 2 upsampling blocks.
*   **Discriminator:** A 70x70 PatchGAN that classifies patches as real or fake.
*   **Loss Functions:**
    1.  **Perceptual Loss ($\mathcal{L}_\text{perc}$):** MSE on VGG-19 feature maps.
    $$ \mathcal{L}_\text{perc} = \sum_{i} \left\| \phi_i\left(G(B)\right) - \phi_i(I) \right\|_2^2 $$
    2.  **Adversarial Loss ($\mathcal{L}_\text{adv}$):** Wasserstein GAN with Gradient Penalty (WGAN-GP).
*   **Total Generator Loss:**
*   \[\mathcal{L}_{G}\lambda_{\text{perc}}\;\mathcal{L}_{\text{perc}}\;+\;\lambda_{\text{adv}}\;\mathcal{L}_{\text{adv}}\]


#### Training Setup
*   **Dataset:** GoPro (2,103 train / 1,111 test pairs).
*   **Batch Size:** 16
*   **Learning Rate:** $2 \times 10^{-4}$ (Adam).
*   **Hardware:** NVIDIA Tesla P100 (≈ 48 hours).

#### Performance
*   **GoPro Test:** PSNR = 29.34 dB, SSIM = 0.929
*   **Parameters:** ~10.4 M
*   **Inference Time:** ~0.041 s/image (256x256)

> **Note:** DeblurGAN runs faster than many contemporary methods while achieving excellent visual quality.

### 3.3 SRN-Deblur (Scale-Recurrent Network)

#### Approach Summary
SRN-Deblur uses a coarse-to-fine, multi-scale recurrent architecture to iteratively deblur an image. By sharing weights across scales, it effectively uses a large receptive field with fewer parameters to handle severe motion blur.

#### Architecture Details
*   **Structure:** A 3-scale pyramid where the output of a coarser scale is fed into the next finer scale.
*   **Core Module:** An Encoder-Decoder network with ResBlocks is shared across all scales.
*   **Recurrency:** A hidden state is passed from coarser to finer scales, guiding the restoration.
*   **Loss Function:** The sum of MSE losses calculated at each scale.
    $$ \mathcal{L} = \sum_{s=1}^S \left\| I_s - \hat{I}_s \right\|_2^2 $$

#### Training Setup
*   **Dataset:** GoPro.
*   **Batch Size:** 16
*   **Optimizer:** Adam, with learning rate decay over 2,000 epochs.
*   **Hardware:** NVIDIA Titan X (≈ 72 hours).

#### Performance
*   **GoPro Test (3-scale):** PSNR = 29.98 dB, SSIM = 0.929
*   **Parameters:** ~5.17 M
*   **Inference Time:** ~0.08 s/image

> **Note:** The multi-scale design with weight sharing is key to its performance and efficiency.

### 3.4 Deblurring by Realistic Blurring (BGAN + DBGAN)

#### Approach Summary
This two-stage approach first trains a **Blur-GAN (BGAN)** to learn realistic blur kernels from unpaired real-world blurry images. Then, a **Deblur-GAN (DBGAN)** is trained on synthetic data generated by the BGAN, allowing it to generalize better to real-world blur.

#### Architecture Details
1.  **BGAN (Blur-GAN):** A CycleGAN-like network trained on unpaired sharp and real blurry images to learn a realistic blur synthesis function $G_B: I \to B_{syn}$.
2.  **DBGAN (Deblur-GAN):** A standard deblurring GAN (similar to DeblurGAN) trained on paired data $(B_{syn}, I)$ generated by BGAN.
*   **Losses:** A combination of adversarial loss, perceptual loss, and a novel relative blur loss to ensure the synthetic blur distribution matches the real one.

#### Training Setup
*   **Datasets:** Sharp images (COCO), real blurry images (RWBI), and synthetic pairs from BGAN.
*   **Batch Size:** 8
*   **Optimizer:** Adam ($lr = 2 \times 10^{-4}$).
*   **Hardware:** NVIDIA Tesla V100 (≈ 48 hours total).

#### Performance
*   **GoPro:** PSNR = 30.12 dB, SSIM = 0.932
*   **RealBlur:** PSNR = 29.87 dB, SSIM = 0.928
*   **Parameters:** ~12.5 M
*   **Inference Time:** ~0.095 s/image

> **Note:** By learning to synthesize realistic blur first, this method achieves superior generalization on real-world test sets.

### 3.5 MPRNet (Multi-Stage Progressive Restoration)

#### Approach Summary
MPRNet decomposes restoration into three progressive stages. The first two stages operate on downsampled feature maps to capture context, while the final stage works at the original resolution to refine details. Cross-Stage Feature Fusion (CSFF) allows information to flow between stages.

#### Architecture Details
*   **Stages 1 & 2:** U-Net style encoder-decoders using **Channel Attention Blocks (CABs)**.
*   **Stage 3:** A subnetwork of **Original Resolution Blocks (ORBs)** that avoids downsampling to preserve high-frequency details.
*   **CSFF:** Features from earlier stage encoders are fused into later stages to maintain a rich feature representation.
*   **Loss Function:** A simple pixel-wise loss (e.g., L1 or L2) on the final output.

#### Training Setup
*   **Dataset:** GoPro (training). Tested on GoPro, HIDE, RealBlur.
*   **Batch Size:** 16 (on 4x V100 GPUs).
*   **Optimizer:** Adam with a cosine annealing schedule.
*   **Hardware:** 4x NVIDIA Tesla V100 (≈ 30 hours).

#### Performance
*   **GoPro:** PSNR = 31.12 dB, SSIM = 0.945
*   **Parameters:** ~21.1 M
*   **Inference Time:** ~0.032 s/image

> **Note:** MPRNet's multi-stage design excels at balancing large-scale context restoration with fine-grained detail preservation.

### 3.6 HINet (Half Instance Normalization Network)

#### Approach Summary
HINet introduces a novel **Half Instance Normalization (HIN) Block** that splits feature channels, applying instance normalization to one half while leaving the other untouched. This balances content preservation and style normalization, leading to state-of-the-art results with high efficiency.

#### Architecture Details
*   **HIN Block:** The core component. Splits channels into two branches—one with Instance Norm (IN) and one without—and concatenates the results. This preserves both stylistic features and image-specific details.
*   **Overall Structure:** A multi-stage architecture similar to MPRNet, but replacing standard attention blocks with HIN Blocks.
    *   **Stages 1 & 2:** U-Net style encoder-decoders with HIN Blocks.
    *   **Stage 3:** Original resolution subnetwork with HIN Blocks.
*   **Loss Function:** L1 loss on the final output.

#### Training Setup
*   **Dataset:** GoPro.
*   **Batch Size:** 16
*   **Optimizer:** Adam ($lr = 4 \times 10^{-4}$).
*   **Hardware:** Single NVIDIA Titan RTX (≈ 36 hours).

#### Performance
*   **GoPro:** PSNR = 31.05 dB, SSIM = 0.944
*   **Parameters:** ~19.8 M
*   **Inference Time:** ~0.025 s/image

> **Note:** The HIN Block provides a powerful yet efficient way to handle the feature statistics, making HINet one of the fastest and most accurate methods.

---

## 4. Performance Comparison Tables

A consolidated comparison across methods on standard deblurring benchmarks. Inference times are measured on modern GPUs for a 256x256 image.

| Method | GoPro (PSNR/SSIM) | HIDE (PSNR/SSIM) | RealBlur (PSNR/SSIM) | Params (M) | FLOPs (G) | Inference (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SRCNN-Based CNN | 27.50 / 0.850 | 26.80 / 0.835 | 26.45 / 0.830 | 0.06 | 35 | 0.012 |
| DeblurGAN | 29.34 / 0.929 | 28.97 / 0.911 | 28.65 / 0.909 | 10.4 | 300 | 0.041 |
| SRN-Deblur | 29.98 / 0.929 | 29.12 / 0.915 | 28.87 / 0.912 | 5.17 | 220 | 0.080 |
| Realistic Blur | 30.12 / 0.932 | 29.45 / 0.920 | **29.98 / 0.933** | 12.5 | 350 | 0.095 |
| **MPRNet** | **31.12 / 0.945** | **30.28 / 0.937** | **29.98 / 0.933** | 21.1 | 260 | 0.032 |
| **HINet** | 31.05 / 0.944 | 30.20 / 0.936 | 29.92 / 0.930 | 19.8 | 210 | **0.025** |

---

## 5. Summary & Practical Recommendations

1.  **For Low-Resource / Fast Prototyping:**
    *   **SRCNN-Based CNN:** Ideal for quick baselines on limited hardware. Simple to implement but limited in performance.

2.  **For Good Visual Quality with Moderate Resources:**
    *   **DeblurGAN:** A great choice if visually pleasing results are more important than peak PSNR. Trains reasonably fast on a single mid-range GPU.

3.  **For a Balanced Resource-Performance Trade-off:**
    *   **SRN-Deblur:** Offers a compact model (~5M params) that effectively handles large motion blur.

4.  **For Best Generalization to Real-World Blur:**
    *   **Realistic Blurring (BGAN + DBGAN):** The best option if your target domain involves diverse, real-world blur and you have access to unpaired real blurry images.

5.  **For Top-Tier Performance (State-of-the-Art):**
    *   **MPRNet:** Delivers the highest PSNR/SSIM on standard benchmarks. The best choice when raw accuracy is the top priority.
    *   **HINet:** Offers performance nearly identical to MPRNet but with fewer parameters and the fastest inference speed among top models. Highly recommended for production environments where both speed and accuracy are critical.

---

## 6. References

1.  Dong, C., Loy, C. C., He, K., & Tang, X. (2015). **"Image Super-Resolution Using Deep Convolutional Networks"** (basis for SRCNN). *IEEE TPAMI*.
2.  Kupyn, O., Budzan, V., Mykhailych, M., Mishkin, D., & Matas, J. (2018). **“DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks”**. *CVPR*.
3.  Tao, X., Gao, H., Shen, X., Wang, J., & Jia, J. (2018). **“Scale-recurrent Network for Deep Image Deblurring”**. *CVPR*.
4.  Zhang, K., Luo, W., Zhong, Y., & Ma, L. (2020). **“Deblurring by Realistic Blurring”**. *CVPR*.
5.  Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., & Yang, M. H. (2021). **“Multi-Stage Progressive Image Restoration”** (MPRNet). *CVPR*.
6.  Chen, L., Lu, X., Zhang, J., Chu, X., & Chen, C. (2021). **“HINet: Half Instance Normalization Network for Image Restoration”**. *CVPR Workshops*.

---

## License
MIT License

Feel free to copy, modify, and redistribute this document. If you find it useful in your work, please consider citing the original papers listed in the References section.
