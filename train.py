import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import os
import glob
from PIL import Image

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siêu tham số
batch_size = 32
image_size = 64
latent_dim = 100
epochs = 50
lr = 0.0002
beta1 = 0.5

data_dir = "F:/TTNT/processed_images"  # Thay đổi đường dẫn nếu cần

# Dataset tùy chỉnh
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = glob.glob(os.path.join(root_dir, "*.pt"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_tensor = torch.load(self.files[idx])
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor

# Biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Bộ sinh (Generator)
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size * image_size * 3),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, image_size, image_size)
        return img

# Bộ phân biệt (Discriminator)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Khởi tạo mô hình
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Huấn luyện
for epoch in range(epochs):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        valid = torch.ones(real_imgs.size(0), 1, device=device)
        fake = torch.zeros(real_imgs.size(0), 1, device=device)

        # Huấn luyện Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z)
        loss_real = criterion(discriminator(real_imgs), valid)
        loss_fake = criterion(discriminator(gen_imgs.detach()), fake)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Huấn luyện Generator
        optimizer_G.zero_grad()
        loss_G = criterion(discriminator(gen_imgs), valid)
        loss_G.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # Lưu ảnh sau mỗi epoch
    vutils.save_image(gen_imgs[:25], f"F:/TTNT/results/epoch_{epoch}.png", normalize=True)

# Lưu mô hình
torch.save(generator.state_dict(), "F:/TTNT/models/generator.pth")
torch.save(discriminator.state_dict(), "F:/TTNT/models/discriminator.pth")

print("✅ Huấn luyện hoàn tất và mô hình đã được lưu!")
