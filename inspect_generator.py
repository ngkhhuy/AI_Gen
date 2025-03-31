import torch

# Đường dẫn đến file checkpoint
checkpoint_path = "F:/TTNT/models/generator.pth"

# Load checkpoint (sử dụng map_location="cpu" nếu không có GPU)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# In ra các key và kích thước tensor
print("Các key trong checkpoint:")
for key, value in checkpoint.items():
    print(f"{key}: {value.shape}")
