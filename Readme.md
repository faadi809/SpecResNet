# SpecResNet: Hyperspectral Image Compression via Hybrid Residual Learning and Spectral Calibration
### Fahad Saeed, Shumin Liu, and Jie Chen

### Abstract
Hyperspectral imaging provides rich spatial–spectral information but generates huge data volumes, posing significant challenges for storage, transmission, and real-time processing in remote sensing applications. In this study, we propose SpecResNet, a 3D autoencoderbased model for hyperspectral image compression. This framework introduces hybrid residual blocks for preserving representational power and a spectral calibration (SC) block to enhance spectral fidelity. It also uses Squeeze-and-Excitation (SE) blocks for adaptive feature recalibration. Our model obtains different compression operating points by varying model capacity, with bitrate emerging implicitly from the learned latent representations. Experiments on several benchmark datasets show that SpecResNet surpasses the performance of existing frameworks on most datasets in terms of PSNR, MS-SSIM, and SAM, demonstrating its strong potential. Our results suggest that SpecResNet offers a promising trade-off for efficient hyperspectral image compression, with potential for further refinement in complex scenes

## Architecture Overview

The model (`SpecResNet`) is a symmetric 3D convolutional autoencoder that processes hyperspectral patches of shape `(N, C, 1, H, W)` where:
- `N` = batch size
- `C` = 48 spectral bands per chunk
- `H, W` = 64×64 spatial patch

**Encoder**: Conv3d → HybridResGroupedBlock3D → SEBlock3D (×4 levels with progressive downsampling)    
**Bottleneck**: GradualQuantization module with progressive noise injection during training

**Decoder**: ConvTranspose3d → HybridResGroupedBlock3D → SEBlock3D (×4 levels with progressive upsampling)  
**Final layer**: SC_Module (Spectral Context) — learnable spectral weighting + channel-wise attention

## Datasets

Seven hyperspectral datasets are used:

| Dataset | File | Key | Shape (H×W×B) |
|---------|------|-----|----------------|
| PaviaU | `PaviaU/PaviaU.mat` | `paviaU` | 610×340×103 |
| PaviaC | `PaviaC/Pavia.mat` | `pavia` | 1096×715×102 |
| Botswana | `Botswana/Botswana.mat` | `Botswana` | 1476×256×145 |
| Houston 2013 | `Houston/Houston13.mat` | `data` | 349×1905×144 |
| Houston 2018 | `Houston/Houston2018_input.mat` | `input` | 601×2384×48 |
| Chikusei | `Chikusei/Chikusei.mat` | `chikusei` | 2517×2335×128 (via h5py) |
| Washington DC | `WashingtonDC/DC.tif` | `dc` | 1208×307×191 |

Update paths in `config.py` to match your local filesystem.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

Edit `config.py` to set dataset paths, GPU IDs, and hyperparameters.

### Evaluation

```bash
python evaluate.py
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| `in_channels` | 48 (spectral chunk size) |
| `base_channels` | 256 |
| `patch_size` | (64, 64, 48) |
| `stride` | (32, 32) |
| `batch_size` | 96 |
| `learning_rate` | 0.0005 |
| `lambda_tv` | 0.15 |
| `lambda_sam` | 0.26 |
| `num_epochs` | 300 |
| `patience` | 10 |

## Loss Function

Total loss = **Charbonnier** + `lambda_tv` × **TV Loss** + `lambda_sam` × **SAM Loss**

- **Charbonnier**: `mean(sqrt((pred−target)^2 + eps^2))`, `eps=1e-3`
- **TV Loss**: 3D total variation across spatial and spectral dimensions
- **SAM Loss**: Spectral Angle Mapper — mean spectral angle between predicted and target spectra

## Data Splitting

| Dataset | Train Split | Test Split |
|---------|------------|------------|
| PaviaU, PaviaC, Houston13, Botswana, Houston18, DC, Chikusei | First 70% rows | Last 30% rows |

## Spectral Chunking

During inference, each hyperspectral cube is processed in spectral chunks of 48 bands:
- If the last chunk has fewer than 48 bands, it is **zero-padded** to 48 bands
- The zero-padded bands are **trimmed** from the reconstruction output

### Model Loading
The saved checkpoint is a **dict** (not a bare `state_dict`). Load as:
```python
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
```

### GPU Setup
Set `CUDA_VISIBLE_DEVICES` in `config.py` **before** any `torch.cuda` calls. The model automatically uses `nn.DataParallel` when multiple GPUs are available.

### Entropy Coding

Compression metrics use `compressai.entropy_models.EntropyBottleneck` with `channels` = `base_channels // 4`) and is adjusted for different bitrates.

### To cite:
@article{Saeed2026SpecResNet,
  author  = {Fahad Saeed and Shumin Liu and Jie Chen},
  title   = {SpecResNet: Hyperspectral Image Compression via Hybrid Residual Learning and Spectral Calibration},
  journal = {Remote Sensing},
  year    = {2026},
  month   = {April},
}


