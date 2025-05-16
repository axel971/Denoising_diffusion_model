import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DiffusionModel():

    def __init__(self, T : int, model : nn.Module, device : str):

        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device) # Beta is fixed (in therory could be also trained)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training_step(self,
                      dataloader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: nn.Module, #MSE loss
                      device: torch.device
                      ):
        """

        Training of the Diffusion model for one epoch

        """

        loss_value = 0
        self.function_approximator.train()
        
        loop = tqdm(dataloader)
        for (x0, _) in loop:

            x0 = x0.to(device) # To do: check the size of x0
            t = torch.randint(1, self.T + 1, (x0.shape[0],), device=self.device, dtype=torch.long)
            eps = torch.randn_like(x0) # To do: check the size of eps

            alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #print(alpha_bar_t.shape)
            #print(x0.shape)
            eps_predicted = self.function_approximator(torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t-1)
        
            loss = loss_fn(eps, eps_predicted)
            loss_value += loss.item()
            
            loop.set_postfix(loss = loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_value = loss_value/len(dataloader)

        return loss_value

    
    def train(self,
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim,
              loss_fn: nn.Module,
              epochs: int,
              device: torch.device
              ):


        results = { "training_loss": []}

        for epoch in range(epochs):

            print("Epoch: {epoch + 1}/{epochs}: ")
            
            loss_value = self.training_step(dataloader = dataloader,
                                        optimizer = optimizer,
                                        loss_fn = loss_fn,
                                        device = device
                                        )
            
            results["training_loss"].append(loss_value)


        return results

    
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), use_tqdm=True):

        """
        sampling is a function that generates synthetic images from a tensor of noisy images (using the reverse diffusion process)
        """


        self.function_approximator.eval()
        with torch.inference_mode():

            x = torch.randn((n_samples,
                             image_channels,
                             img_size[0],
                             img_size[1])
                            , device=self.device) # x is a tensor of noisy images (begining of the reverse process)

            progress_bar = tqdm if use_tqdm else lambda x : x

            for t in progress_bar(range(self.T, 0, -1)):

                z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                
                t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t
                
                beta_t = self.beta[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                alpha_t = self.alpha[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * self.function_approximator(x, t-1))

                sigma = torch.sqrt(beta_t)

                x = mean + sigma * z

        return x


