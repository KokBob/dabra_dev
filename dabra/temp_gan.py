# Imports
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import pandas as pd
import numpy as np
import math


# Settings
torch.manual_seed(111)
lr = 0.0001
num_epochs = 50
batch_size = 2
# possible batch sizes: 2,5,10,17,34,37,74,85,170 for a dataset of 6290 images
# possible batch sizes: {1,2,3,4,6,9,12,18,36,101,202,303,404,606,909,1212,1818,3636} for a dataset of 3636 images
loss_function = nn.BCELoss()  # binary cross-entropy loss
lat_sample_size = 1000
save_models = True
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Class definitions
class BackgroundImages(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        image_list = os.listdir('./image_extraction/backgrounds/')
        self.img_labels = pd.DataFrame(np.ones((len(image_list), 1)))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_list = os.listdir('./image_extraction/backgrounds/')
        img_path = "./image_extraction/backgrounds/"+image_list[idx]
        image = read_image(img_path, ImageReadMode.GRAY)
        label = torch.tensor(1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(300*300, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 300*300)  # Vectorize image batch (batch_sizex1x300x300) --> (1x90000)
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(lat_sample_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 300*300),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 300, 300)
        return output

if __name__ == '__main__':

    # Instantiate Discriminator and Generator
    discriminator = Discriminator().to(device=device)
    generator = Generator().to(device=device)

    # Prepare background images as training data
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]   # function to be used when loading the data
    )
    training_data = BackgroundImages(img_dir='./image_extraction/backgrounds/', transform=transform)
    train_loader = DataLoader(training_data, batch_size, shuffle=True)

    # plot some examples
    batch_example = next(iter(train_loader))
    for i in range(0, batch_size):
        plt.imshow(batch_example[0][i].squeeze(), cmap="gray")
        plt.title("test data sample #: "+str(i+1))
        plt.show()

    # Training the models
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for n, (real_samples, labels) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            real_samples_labels = torch.ones((batch_size, 1)).to(device=device)

            latent_space_samples = torch.randn((batch_size, lat_sample_size)).to(device=device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)

            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # Training the Discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            test = output_discriminator.size()
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, lat_sample_size)).to(device=device)

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if n == batch_size-1:
                print(f"Epoch:{epoch} Loss Discriminator: {loss_discriminator}")
                print(f"Epoch:{epoch} Loss Generator: {loss_generator}")

        if ((epoch+1) % 1) == 0:
            # Checking the samples generated by the GAN after each epoch
            temp_input = torch.randn((16, lat_sample_size)).to(device=device)
            plot_samples = generator(temp_input)
            plot_samples = plot_samples.cpu().detach()
            for i in range(16):
                ax = plt.subplot(4, 4, i + 1)
                plt.imshow(plot_samples[i].squeeze(), cmap="gray")
                plt.xticks([])
                plt.yticks([])

            plt.suptitle("generated samples after epoch: " + str(epoch+1))
            plt.show()


if save_models:
    # save models
    dir_path = os.path.dirname(os.path.realpath(__file__))
    torch.save(generator, dir_path+"/generator_backgroundGANtemp.pt")
    torch.save(discriminator, dir_path+"/discriminator_backgroundGANtemp.pt")


