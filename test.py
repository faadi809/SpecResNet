import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from compressai.entropy_models import EntropyBottleneck
from models.specresnet import Autoencoder3D
from utils.dataset import load_h5_cube, normalize, spatial_split


# -------------------------------------------------
# Load trained model (inference only)
# -------------------------------------------------
def load_model(weights_path, device):
    model = Autoencoder3D(in_channels=48, base_channels=256)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------
# Initialize entropy bottleneck (inference only)
# -------------------------------------------------
def init_entropy_bottleneck(channels, device):
    eb = EntropyBottleneck(channels=channels).to(device)
    eb.eval()
    eb.update()
    return eb


# -------------------------------------------------
# Spectral chunking + compression + reconstruction
# -------------------------------------------------
def chunk_spectral_cube(
    cube_hw_b,
    model,
    entropy_bottleneck,
    chunk_size=48,
    device="cuda"
):
    if isinstance(cube_hw_b, np.ndarray):
        cube_hw_b = torch.from_numpy(cube_hw_b).float()

    cube_hw_b = cube_hw_b.clamp(0.0, 1.0)

    # unwrap DataParallel if any
    while hasattr(model, "module"):
        model = model.module

    H, W, B = cube_hw_b.shape
    recon_chunks = []
    latent_chunks = []
    compressed_chunks = []

    with torch.no_grad():
        for i in range(0, B, chunk_size):
            chunk = cube_hw_b[:, :, i:i + chunk_size]
            current_bands = chunk.shape[2]

            if current_bands < chunk_size:
                pad = chunk_size - current_bands
                chunk = F.pad(chunk, (0, pad), mode="constant", value=0)

            inp = (
                chunk.permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(2)
                .to(device)
            )  # [1, C, 1, H, W]

            _, latent = model(inp)
            latent = latent.squeeze(2) + 1e-8  # [1, C, H, W]

            compressed = entropy_bottleneck.compress(latent)
            compressed_chunks.append(compressed)

            shape = latent.shape[-2:]
            try:
                latent_hat = entropy_bottleneck.decompress(compressed, shape)
            except Exception:
                latent_hat = latent.clone()

            # Decoder forward
            x = latent_hat.unsqueeze(2)
            x = model.dec01(x)
            x = model.dec1(x)
            x = model.dec11(x)
            x = model.dec2(x)
            x = model.final(x)

            recon = (
                x.squeeze(0)
                .squeeze(1)
                .permute(1, 2, 0)
                .cpu()
            )

            recon = recon[:, :, :current_bands]
            latent = latent[:, :, :current_bands].cpu()

            recon_chunks.append(recon)
            latent_chunks.append(latent)

    recon_cube = torch.cat(recon_chunks, dim=2)
    latent_cube = torch.cat(latent_chunks, dim=2)

    return recon_cube, latent_cube, compressed_chunks


# -------------------------------------------------
# Bitrate calculation (bpppb)
# -------------------------------------------------
def calculate_bitrate(compressed_stream, original_shape):
    total_bytes = sum(len(b) for chunk in compressed_stream for b in chunk)
    total_bits = total_bytes * 8
    H, W, B = original_shape
    return total_bits / (H * W * B)


# -------------------------------------------------
# Main evaluation entry
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "/home/data/Fahad/models/My3d_256_Bots2_big_bottleneck_LR0.0005(patch64)test5.pth"
    chikusei_path = "/home/data/Fahad/HSI_datasets/Chikusei/Chikusei.mat"

    model = load_model(model_path, device)
    entropy_bottleneck = init_entropy_bottleneck(channels=64, device=device)

   # cube = load_h5_cube(chikusei_path, "chikusei")
    #cube = normalize(cube).transpose(1, 2, 0)
   # _, test_cube = spatial_split(cube, test_ratio=0.3)

    test_cube = test_cube[:128, :128, :48] 

    recon_cube, latent_cube, compressed = chunk_spectral_cube(
        test_cube,
        model,
        entropy_bottleneck,
        chunk_size=48,
        device=device,
    )

    bitrate = calculate_bitrate(compressed, test_cube.shape)

    print(f"Reconstructed cube shape: {recon_cube.shape}")
    print(f"Bitrate: {bitrate:.4f} bpppb")


if __name__ == "__main__":
    main()
