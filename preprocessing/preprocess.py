import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Đường dẫn đến dataset raw
RAW_DATASET_DIR = "TTNT/dataset/archive/raw-img"
PROCESSED_DATASET_DIR = "TTNT/processed_images"

# Tạo thư mục lưu ảnh đã xử lý nếu chưa có
os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)

# Định nghĩa các biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize về 64x64
    transforms.ToTensor(),        # Chuyển thành tensor
    transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa về [-1,1]
])

# Load dataset
dataset = ImageFolder(root=RAW_DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Lưu ảnh đã xử lý dưới dạng tensor
for i, (images, _) in enumerate(dataloader):
    for j, image in enumerate(images):
        torch.save(image, os.path.join(PROCESSED_DATASET_DIR, f"image_{i*32+j}.pt"))

print(f"✅ Đã tiền xử lý xong {len(dataset)} ảnh và lưu vào {PROCESSED_DATASET_DIR}")
