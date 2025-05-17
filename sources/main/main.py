
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
from pathlib import Path 

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from models.unet import UNet
from models.DiffusionModel import DiffusionModel

def main(data_dir: str):

    DATA_DIR = Path(data_dir)

    # Create transform
    transform = transforms.Compose([
            transforms.Resize(size = (32, 32)),
            transforms.ToTensor()]
            )

    # Load the data
    training_data = datasets.MNIST(
            root = DATA_DIR,
            train = True,
            download = True,
            transform = transform
            )

    testing_data = datasets.MNIST(
            root = DATA_DIR,
            train = False,
            download = True,
            transform = transform
            )

    # Create dataloader
    BATCH_SIZE = 64

    training_dataloader = DataLoader(dataset = training_data,
                                     batch_size = BATCH_SIZE,
                                     num_workers = 3,
                                     shuffle = True)

    testing_dataloader = DataLoader(dataset = testing_data,
                                    batch_size = BATCH_SIZE,
                                    num_workers = 3,
                                    shuffle = False)

    # Device agnostic code
    device = "gpu " if torch.cuda.is_available() else "cpu"


    # Initialize the models
    model = UNet()
    diffusion_model = DiffusionModel(1000, model, device)

    # Initialize the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_fn = nn.MSELoss()

    # Train the network
    EPOCHS = 4

    results = diffusion_model.train(dataloader = training_dataloader,
                          loss_fn = loss_fn,
                          optimizer = optimizer,
                          epochs = EPOCHS,
                          device = device)

    # Create samples
    n_samples = 81
    
    samples = diffusion_model.sampling(n_samples = n_samples)
    plt.figure(figsize = (17, 17))

    for i in range(n_samples):
        plt.subplot(9, 9, i + 1)
        plt.axis("off")
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap = "gray")
    plt.savefig(f"samples_epoch_{EPOCHS}.png")
    plt.close()
    
    # Display loss function
    plt.plot(results["training_loss"])
    plt.savefig("training_loss.png")
    plt.close()

    """
for epoch in tqdm(range(40_000)):
    loss = diffusion_model.training(batch_size, optimizer)
    training_loss.append(loss)

    if epoch % 100 == 0:
        plt.plot(training_loss)
        plt.savefig('training_loss.png')
        plt.close()

        plt.plot(training_loss[-1000:])
        plt.savefig('training_loss_cropped.png')
        plt.close()
    """
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required = True, help = "Path toward data directory")
    args = parser.parse_args()

    main(data_dir = args.data_dir)
