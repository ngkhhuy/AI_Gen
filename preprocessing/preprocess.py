import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Đường dẫn dataset
RAW_DATASET_DIR = os.path.join("F:\\", "TTNT", "dataset", "archive", "raw-img")
PROCESSED_DATASET_DIR = os.path.join("F:\\", "TTNT", "processed_images")

# Kiểm tra thư mục dataset
if not os.path.exists(RAW_DATASET_DIR):
    raise FileNotFoundError(f"Thư mục dataset không tồn tại: {RAW_DATASET_DIR}")

print(f"Đang xử lý dữ liệu từ: {RAW_DATASET_DIR}")
print(f"Sẽ lưu kết quả vào: {PROCESSED_DATASET_DIR}")

# Tạo thư mục lưu ảnh đã xử lý nếu chưa có
os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)

# Định nghĩa các biến đổi nâng cao cho ảnh
transform = transforms.Compose([
    # Chuyển đổi sang PIL Image nếu chưa phải
    transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else Image.fromarray(x)),
    
    # Cân bằng màu sắc và tăng cường chất lượng
    transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(1.2)),  # Tăng độ tương phản
    transforms.Lambda(lambda img: ImageEnhance.Brightness(img).enhance(1.1)),  # Tăng độ sáng nhẹ
    transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(1.3)),  # Tăng độ sắc nét
    
    # Áp dụng bộ lọc làm mịn nhẹ để giảm nhiễu
    transforms.Lambda(lambda img: img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))),
    
    # Resize với chất lượng cao hơn
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS),
    
    # Chuẩn hóa dữ liệu
    transforms.ToTensor(),
    
    # Chuẩn hóa với giá trị trung bình và độ lệch chuẩn tốt hơn cho hình ảnh tự nhiên
    # Sử dụng giá trị chuẩn của ImageNet để có kết quả tốt hơn
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hàm kiểm tra và cải thiện chất lượng ảnh
def enhance_image_quality(image_tensor):
    # Chuyển tensor về khoảng [0, 1] để xử lý
    image = image_tensor.clone()
    
    # Đảm bảo ảnh có 3 kênh màu
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3, :, :]
    
    # Đảm bảo kích thước ảnh là 64x64
    if image.shape[1] != 64 or image.shape[2] != 64:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), 
            size=(64, 64), 
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
    
    # Kiểm tra và điều chỉnh độ tương phản
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val < 0.5:  # Nếu độ tương phản thấp
        # Tăng độ tương phản
        image = (image - min_val) / (max_val - min_val + 1e-8)
    
    return image

# Load dataset
print("Đang tải dataset...")
dataset = ImageFolder(root=RAW_DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Kiểm tra kích thước ảnh đầu vào
print(f"Số lượng ảnh trong dataset: {len(dataset)}")

# Lấy một batch để kiểm tra
print("Kiểm tra kích thước ảnh đầu vào...")
for images, _ in dataloader:
    print(f"Kích thước batch: {images.shape}")  # Kỳ vọng: [32, 3, 64, 64]
    
    # Kiểm tra giá trị min, max
    print(f"Giá trị min: {images.min():.4f}, Giá trị max: {images.max():.4f}")
    
    # Hiển thị một vài ảnh để kiểm tra
    print("Lưu một số ảnh mẫu để kiểm tra...")
    plt.figure(figsize=(15, 5))
    
    # Hiển thị ảnh gốc
    for i in range(min(5, images.size(0))):
        plt.subplot(2, 5, i+1)
        # Chuyển từ tensor về ảnh để hiển thị (denormalize với giá trị ImageNet)
        img = images[i].clone()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title("Gốc")
        plt.axis('off')
    
    # Hiển thị ảnh đã cải thiện
    for i in range(min(5, images.size(0))):
        plt.subplot(2, 5, i+6)
        # Cải thiện chất lượng ảnh
        enhanced_img = enhance_image_quality(images[i])
        # Denormalize
        enhanced_img = enhanced_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        enhanced_img = enhanced_img.clamp(0, 1)
        enhanced_img = enhanced_img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(enhanced_img)
        plt.title("Đã cải thiện")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DATASET_DIR, "sample_comparison.png"))
    print(f"Đã lưu ảnh so sánh tại: {os.path.join(PROCESSED_DATASET_DIR, 'sample_comparison.png')}")
    break

# Lưu ảnh đã xử lý dưới dạng tensor
print("Bắt đầu xử lý và lưu ảnh...")
count = 0
total_images = len(dataset)

# Tạo thanh tiến trình cho các batch
for i, (images, _) in enumerate(tqdm(dataloader, desc="Xử lý batch")):
    # Tạo thanh tiến trình cho từng ảnh trong batch
    for j, image in enumerate(tqdm(images, desc=f"Batch {i+1}/{len(dataloader)}", leave=False)):
        # Cải thiện chất lượng ảnh
        enhanced_image = enhance_image_quality(image)
        
        # Lưu tensor
        torch.save(enhanced_image, os.path.join(PROCESSED_DATASET_DIR, f"image_{count}.pt"))
        count += 1
        
        # Hiển thị tiến trình tổng thể
        if count % 100 == 0:
            print(f"Đã xử lý {count}/{total_images} ảnh ({count/total_images*100:.1f}%)")

# Tạo một tệp thông tin về dữ liệu đã xử lý
with open(os.path.join(PROCESSED_DATASET_DIR, "dataset_info.txt"), "w", encoding="utf-8") as f:
    f.write(f"Tổng số ảnh: {count}\n")
    f.write(f"Kích thước ảnh: [3, 64, 64]\n")
    f.write(f"Chuẩn hóa: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]\n")
    f.write(f"Các biến đổi đã áp dụng: Tăng độ tương phản, độ sáng, độ sắc nét, làm mịn, resize với Lanczos\n")

print(f"✅ Đã tiền xử lý xong {count} ảnh và lưu vào {PROCESSED_DATASET_DIR}")
print(f"Kích thước ảnh đã xử lý: [3, 64, 64]")
print(f"Đã tạo file thông tin dataset tại: {os.path.join(PROCESSED_DATASET_DIR, 'dataset_info.txt')}")
