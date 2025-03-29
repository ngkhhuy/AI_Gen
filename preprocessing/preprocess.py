import os
import json
import re
from pycocotools.coco import COCO

# Đường dẫn dataset
DATASET_DIR = "F:/TTNT/dataset"
ANNOTATION_FILE = os.path.join(DATASET_DIR, "annotations_trainval2017/annotations/captions_train2017.json")
IMAGE_DIR = os.path.join(DATASET_DIR, "train2017")

# Kiểm tra file có tồn tại không
if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(f"❌ Không tìm thấy file: {ANNOTATION_FILE}")

print("✅ Tệp annotation tồn tại, bắt đầu load dữ liệu...")

# Load COCO dataset
coco = COCO(ANNOTATION_FILE)

# Lấy danh sách ID của hình ảnh
image_ids = coco.getImgIds()

# Ánh xạ hình ảnh với chú thích
image_caption_mapping = {}

for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    captions = [ann['caption'] for ann in annotations]
    image_caption_mapping[img_info['file_name']] = captions

print(f"✅ Đã tải {len(image_caption_mapping)} ảnh và chú thích.")

def clean_caption(caption):
    caption = caption.lower()  # Chuyển thành chữ thường
    caption = re.sub(r"[^a-zA-Z\s]", "", caption)  # Xóa ký tự đặc biệt
    caption = re.sub(r"\s+", " ", caption).strip()  # Xóa khoảng trắng thừa
    return caption

# Áp dụng tiền xử lý lên toàn bộ chú thích
cleaned_captions = {img: [clean_caption(cap) for cap in captions] for img, captions in image_caption_mapping.items()}

# Lưu lại file sau khi làm sạch
CLEANED_OUTPUT_FILE = os.path.join(DATASET_DIR, "annotations_trainval2017", "cleaned_captions.json")
with open(CLEANED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_captions, f, indent=4, ensure_ascii=False)

print(f"✅ Dữ liệu đã được làm sạch và lưu tại: {CLEANED_OUTPUT_FILE}")
