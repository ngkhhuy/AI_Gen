import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from models.dcgan import Generator, Discriminator
from preprocessing.dataset import COCODataset

# Hyperparameters
batch_size = 16
lr = 0.0002
noise_dim = 100
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = COCODataset("F:/TTNT/dataset/processed_images")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Khởi tạo mô hình
G = Generator(noise_dim=noise_dim).to(device)
D = Discriminator().to(device)

# Loss và Optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Huấn luyện
for epoch in range(epochs):
    for i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Tạo nhãn thật và giả
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # *** Huấn luyện Discriminator ***
        optimizer_D.zero_grad()
        outputs = D(real_images)
        loss_real = criterion(outputs, real_labels)

        noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
        fake_images = G(noise)
        outputs = D(fake_images.detach())
        loss_fake = criterion(outputs, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # *** Huấn luyện Generator ***
        optimizer_G.zero_grad()
        outputs = D(fake_images)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], "
                  f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # Lưu checkpoint sau mỗi epoch
    torch.save(G.state_dict(), f"models/generator_epoch_{epoch}.pth")
    torch.save(D.state_dict(), f"models/discriminator_epoch_{epoch}.pth")

    # Lưu ảnh mẫu
    fake_images = fake_images[:16].detach().cpu()
    torch.save(fake_images, f"results/sample_epoch_{epoch}.pt")

print("Huấn luyện hoàn tất!")
