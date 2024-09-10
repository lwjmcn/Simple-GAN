import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# import MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# download MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# dataloader imports dataset in batch
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# function that prints several images
def show_images(images, labels):
    # convert to (-1 ~ 1) to print image
    images = images / 2 + 0.5
    # convert image to numpy array 
    npimg = images.numpy() 
    fig, axes = plt.subplots(1, 8, figsize=(12, 12))

    # print image
    for i in range(8):
        ax = axes[i]
        ax.imshow(npimg[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.show()

# import image array
dataiter = iter(dataloader)
images, labels = next(dataiter)

show_images(images, labels)


# Generator Neural Network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # MLP (Multi-Layer Perceptron)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Neural Network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(), # activation function
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1), # 1D scalar real or fake 
            nn.Sigmoid() # probability
        )

    def forward(self, x):
        return self.model(x)


# determine hyperparamter
latent_dim = 100 # (generator input noise dimension)
img_shape = 28 * 28 

generator = Generator(input_dim=latent_dim, output_dim=img_shape)
discriminator = Discriminator(input_dim=img_shape)

lr = 0.0002 # learning rate
b1 = 0.5 
b2 = 0.999

# optimizer of each
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Loss function
adversarial_loss = nn.BCELoss() 


# execute
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# every epoch, training
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # create a label (valid or fake)
        valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(device)
        fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(device)
        
        # real image from dataset
        real_imgs = imgs.view(imgs.size(0), -1).to(device)
        

        # initialize grad for generator 
        optimizer_G.zero_grad()

        # random noise
        z = torch.randn((imgs.size(0), latent_dim)).to(device)
        # generate image
        gen_imgs = generator(z)
        
        # put a fake image and real label to loss function 
        # if discriminator is fooled, g_loss decreases
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        # backpropagation
        g_loss.backward()
        optimizer_G.step()
        

        # initialize grad for discriminator
        optimizer_D.zero_grad()
        
        # determine real image and calculate d_loss
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        # backpropagation
        d_loss.backward()
        optimizer_D.step()
        
        if i % 400 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")
            
    # save created image to check in middle
    if epoch % 10 == 0:
        gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
        save_image(gen_imgs.data[:25], f"images/{epoch}.png", nrow=5, normalize=True)


# print created image
def show_generated_imgs(generator, latent_dim, num_images=5):
    z = torch.randn(num_images, latent_dim).to(device)
    gen_imgs = generator(z)
    gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
    gen_imgs = gen_imgs.detach().cpu().numpy()

    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    for i in range(num_images):
        axes[i].imshow(np.transpose(gen_imgs[i], (1, 2, 0)).squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.show()

show_generated_imgs(generator, latent_dim)
