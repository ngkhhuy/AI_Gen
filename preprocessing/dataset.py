import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_files = [f for f in sorted(os.listdir(tensor_dir)) if f.endswith('.pt')]
        self.tensor_dir = tensor_dir
        # Backup transform nếu cần tải lại ảnh gốc
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, self.tensor_files[idx])
        try:
            # Thử tải tensor
            tensor = torch.load(tensor_path)
            return tensor
        except Exception as e:
            print(f"Lỗi khi tải file {tensor_path}: {e}")
            # Nếu có lỗi, trả về tensor ngẫu nhiên với kích thước phù hợp
            return torch.randn(3, 256, 256)

# Test Dataset
if __name__ == "__main__":
    dataset = COCODataset("F:/TTNT/dataset/processed_images")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Tổng số ảnh: {len(dataset)}")
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        break
