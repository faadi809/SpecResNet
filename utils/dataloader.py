import torch
from torch.utils.data import Dataset, DataLoader, random_split


class HyperspectralPatchDataset(Dataset):
    def __init__(self, patches):
        # patches: (N, H, W, C)
        patches = torch.from_numpy(patches)
        patches = patches.permute(0, 3, 1, 2)  # (N, C, H, W)
        self.data = patches.unsqueeze(2)       # (N, C, 1, H, W)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders(patches, batch_size=128, split_ratio=0.8, num_workers=8):
    dataset = HyperspectralPatchDataset(patches)
    n_train = int(len(dataset) * split_ratio)
    n_val = len(dataset) - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
