import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import os

from models.dcgan import DCGANGenerator, DCGANDiscriminator

# Tham số huấn luyện
batch_size = 32
image_size = 64
latent_dim = 100
num_epochs = 50
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load dataset (ví dụ: ImageFolder)
data_root = "F:/TTNT/dataset/archive/raw-img"

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # RGB => 3 kênh => (0.5, 0.5, 0.5)
])
dataset = dsets.ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Khởi tạo mô hình
G = DCGANGenerator(latent_dim=latent_dim).to(device)
D = DCGANDiscriminator().to(device)

# 3. Định nghĩa hàm mất mát & Optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# 4. Vòng lặp huấn luyện
os.makedirs("F:/TTNT/results_dcgan", exist_ok=True)
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size_now = real_imgs.size(0)

        # Tạo nhãn thật và giả
        real_labels = torch.ones(batch_size_now, 1, device=device)
        fake_labels = torch.zeros(batch_size_now, 1, device=device)

        # Huấn luyện Discriminator
        optimizer_D.zero_grad()

        # Dữ liệu thật
        outputs = D(real_imgs)
        loss_real = criterion(outputs, real_labels)

        # Dữ liệu giả
        z = torch.randn(batch_size_now, latent_dim, 1, 1, device=device)
        fake_imgs = G(z)
        outputs = D(fake_imgs.detach())
        loss_fake = criterion(outputs, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Huấn luyện Generator
        optimizer_G.zero_grad()
        outputs = D(fake_imgs)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # Lưu ảnh sau mỗi epoch
    vutils.save_image(fake_imgs[:25], f"F:/TTNT/results_dcgan/fake_epoch_{epoch}.png", normalize=True)

# Lưu mô hình
os.makedirs("F:/TTNT/models_dcgan", exist_ok=True)
torch.save(G.state_dict(), "F:/TTNT/models_dcgan/generator_dcgan.pth")
torch.save(D.state_dict(), "F:/TTNT/models_dcgan/discriminator_dcgan.pth")

print("✅ Huấn luyện DCGAN hoàn tất!")
