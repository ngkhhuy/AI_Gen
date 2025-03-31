import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models.dcgan import Generator  # Import Generator từ models/dcgan.py

latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình Generator
generator = Generator(latent_dim=latent_dim).to(device)
generator.load_state_dict(torch.load("F:/TTNT/models/generator.pth", map_location=device))
generator.eval()

# Sinh ảnh mới từ vector nhiễu
num_samples = 16
z = torch.randn(num_samples, latent_dim, device=device)
with torch.no_grad():
    gen_imgs = generator(z)

# Hiển thị ảnh dưới dạng grid
grid = vutils.make_grid(gen_imgs, normalize=True, nrow=4)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.title("Ảnh sinh ra từ DCGAN")
plt.show()

# Lưu ảnh ra file
import os
os.makedirs("F:/TTNT/results", exist_ok=True)
vutils.save_image(gen_imgs, "F:/TTNT/results/generated_samples.png", normalize=True)
print("Ảnh đã được lưu tại F:/TTNT/results/generated_samples.png")
