import torch
import os

tensor_dir = "F:/TTNT/dataset/processed_images"

# Lặp qua tất cả các file .pt
for file in os.listdir(tensor_dir):
    file_path = os.path.join(tensor_dir, file)
    
    try:
        # Thử load file
        torch.load(file_path)
    except Exception as e:
        print(f"❌ Lỗi khi tải file {file_path}: {e}")
        os.remove(file_path)  # Xóa file lỗi
        print(f"🗑️ Đã xóa file {file}")
