# Simple-GAN
2024 HAI Summer Project

# GAN (Generative Adversarial Network)

simple generative model, trains 2 neural networks (Generator, Discriminator) to compete against each other 

- Generator: initial data set, random noise input → generate a new data (fake image)
- Discriminator: real image, fake image input → determine fake or real

![GAN loss function](https://github.com/user-attachments/assets/34a937d1-ad0f-4a71-b535-5a7b1616f963)

GAN loss function

$\it\small\color{#5E5E5E}{(Ian Goodfellow\ 2014)}$

### What can it do?

- create realistic images through text-based prompts/by modifying existing images
- convert a low-resolution image to a high resolution
- turn a black-and-white image to color
- generate 3D models from 2D photos

### Process

1. Generator analyzes the training set and identifies data attributes
2. Discriminator analyzes the initial training data and distinguishes between the attributes independently
3. Generator modifies some data attributes by adding noise
4. Generator passes the modified data to Discriminator
5. Discriminator calculates the probability that the generated output belongs to the original dataset
6. Discriminator gives some guidance to the generator to reduce the noise vector randomization in the next cycle
7. Repeat until equilibrium is reached

![image.png](https://github.com/user-attachments/assets/421a85f7-531b-4d41-858f-ff3374ea36d4)

### Types

- Vanilla GAN
- Conditional GAN (CGAN): The generator and discriminator receive additional info (condition)
- Deep convolutional GAN (DCGAN): integrate CNN architectures, more stable
- Super resolution GAN (SRGAN): upscale low-resolution images to high resolution, Laplacian Pyramid GAN (LAPGAN) breaks down the problem into stages
- Other GAN Models
    - PGGAN: add layers while training from 4x4 resolution to 1024-1024 resolution, good for high-resolution images
    - StyleGAN: based on PGGAN, WGAN-GP for discriminator
    - CycleGAN
    - DiscoGAN
    - WGAN
- Other Image Generation Models
    - DDPM
    - VQ-VAE
  
$\it\small\color{#5E5E5E}{(will\ be\ added\ after)}$

### Reference

https://arxiv.org/abs/1406.2661

https://aws.amazon.com/what-is/gan/?nc2=h_mo-lang



### Execution

```python
pip install torch torchvision numpy matplotlib
```
![image.png](https://github.com/user-attachments/assets/02e8ce58-8742-4918-b070-6d61f7aaffc8)
