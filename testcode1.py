from models.specresnet import Autoencoder3D

model = Autoencoder3D(in_channels=48, base_channels=256)
print(sum(p.numel() for p in model.parameters()))