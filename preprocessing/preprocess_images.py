import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import shutil

# Định nghĩa đường dẫn thư mục ảnh
DATASET_DIR = "F:/TTNT/dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "train2017", "train2017")
OUTPUT_DIR = os.path.join(DATASET_DIR, "processed_images")

# Xóa và tạo lại thư mục đích để đảm bảo sạch sẽ
if os.path.exists(OUTPUT_DIR):
    print(f"Xóa thư mục cũ: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

# Tạo thư mục mới
print(f"Tạo thư mục mới: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kiểm tra quyền ghi
try:
    test_file = os.path.join(OUTPUT_DIR, "test_write.txt")
    with open(test_file, 'w') as f:
        f.write("Test write permission")
    os.remove(test_file)
    print("✅ Kiểm tra quyền ghi: OK")
except Exception as e:
    print(f"❌ Không thể ghi vào thư mục {OUTPUT_DIR}: {e}")
    print("Vui lòng chạy chương trình với quyền admin hoặc kiểm tra lại đường dẫn")
    exit(1)

# Kích thước ảnh đầu ra
IMAGE_SIZE = (256, 256)

# Định nghĩa các phép biến đổi
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # Resize ảnh
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa pixel [-1,1]
])

# Duyệt qua toàn bộ ảnh trong thư mục
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg') or f.endswith('.png')]
print(f"Tìm thấy {len(image_files)} ảnh để xử lý")

# Kiểm tra dung lượng ổ đĩa
import psutil
disk = psutil.disk_usage('/')
free_gb = disk.free / (1024**3)
print(f"Dung lượng trống: {free_gb:.2f} GB")
if free_gb < 5:  # Nếu ít hơn 5GB
    print("⚠️ Cảnh báo: Ổ đĩa có ít dung lượng trống")

success_count = 0
for img_name in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    output_path = os.path.join(OUTPUT_DIR, img_name.replace(".jpg", ".pt").replace(".png", ".pt"))

    try:
        # Mở ảnh
        img = Image.open(img_path).convert("RGB")

        # Áp dụng các phép biến đổi
        img_tensor = transform(img)

        # Kiểm tra tensor
        if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
            print(f"⚠️ Tensor không hợp lệ cho {img_name}, bỏ qua")
            continue

        # Lưu tensor với xử lý lỗi tốt hơn
        try:
            torch.save(img_tensor, output_path)
            # Kiểm tra xem file đã được tạo thành công chưa
            if not os.path.exists(output_path):
                print(f"⚠️ File không được tạo: {output_path}")
                continue
            success_count += 1
        except Exception as e:
            print(f"⚠️ Lỗi khi lưu tensor cho {img_name}: {e}")
            # Thử lại với tên file ngắn hơn nếu đường dẫn quá dài
            short_name = f"{success_count:06d}.pt"
            short_path = os.path.join(OUTPUT_DIR, short_name)
            try:
                torch.save(img_tensor, short_path)
                print(f"✅ Đã lưu với tên ngắn hơn: {short_name}")
                success_count += 1
            except Exception as e2:
                print(f"⚠️ Vẫn không thể lưu với tên ngắn: {e2}")

    except Exception as e:
        print(f"⚠️ Lỗi xử lý {img_name}: {e}")

print(f"✅ Đã xử lý thành công {success_count}/{len(image_files)} ảnh. Ảnh tensor được lưu tại {OUTPUT_DIR}")

# Kiểm tra các file tensor đã tạo
print("Kiểm tra các file tensor...")
tensor_files = os.listdir(OUTPUT_DIR)
valid_count = 0
for tensor_file in tqdm(tensor_files, desc="Validating tensors"):
    tensor_path = os.path.join(OUTPUT_DIR, tensor_file)
    try:
        tensor = torch.load(tensor_path)
        valid_count += 1
    except Exception as e:
        print(f"⚠️ File tensor không hợp lệ: {tensor_file}, lỗi: {e}")
        # Xóa file không hợp lệ
        try:
            os.remove(tensor_path)
            print(f"  Đã xóa file không hợp lệ: {tensor_file}")
        except Exception as e2:
            print(f"  Không thể xóa file: {e2}")

print(f"✅ Có {valid_count}/{len(tensor_files)} file tensor hợp lệ")
