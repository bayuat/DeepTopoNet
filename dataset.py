import torch
from torch.utils.data import Dataset

class BedTopoDataset(Dataset):
    def __init__(self, inputs, target, radar_mask, patch_size=16, stride=8):  # Stride added for overlapping patches
        self.inputs = torch.tensor(inputs, dtype=torch.float32)  # Shape: (5, H, W)
        self.target = torch.tensor(target, dtype=torch.float32)  # Shape: (H, W)
        self.radar_mask = radar_mask  # Shape: (H, W)
        self.patch_size = patch_size
        self.stride = stride
        self.height, self.width = self.target.shape
        self.patches = self._generate_patches()

    def _generate_patches(self):
        patches = []
        for i in range(0, self.height - self.patch_size + 1, self.stride):
            for j in range(0, self.width - self.patch_size + 1, self.stride):
                patches.append((i, j))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col = self.patches[idx]
        input_patch = self.inputs[:, row:row+self.patch_size, col:col+self.patch_size]
        target_patch = self.target[row:row+self.patch_size, col:col+self.patch_size]
        radar_mask_patch = self.radar_mask[row:row+self.patch_size, col:col+self.patch_size]
        return input_patch, target_patch, radar_mask_patch