import os
import math
import torch
import torchvision

from typing import Optional, List
from tqdm import tqdm
from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid, save_image
from .backbone import UNet
from .diffusion import DenoiseDiffusion

class Experimenter():

    device: torch.device = 'cpu'

    eps_model: UNet
    diffusion: DenoiseDiffusion

    # configs of UNet backbone & denoise diffusion
    n_steps: int = 1_000

    # training hyperparameter
    n_samples: int = 64

    optimizer: Adam

    train_dataset: Dataset
    train_dataloader: DataLoader
    eval_dataset: Dataset
    eval_dataloader: DataLoader


    def __init__(
        self,
        dataset_name: str,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        image_channels : int = 3,
        image_size: int = 32,
        n_channels: int = 64,
        channel_multiplers: List[int] = [1, 2, 2, 2],
        is_attention: List[bool] = [False, False, True, True],
        batch_size: int = 128,
        learning_rate: float = 2e-4
    ):
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.image_channels = image_channels
        self.image_size = image_size

        self.eps_model = UNet(
            image_channels=image_channels,
            n_channels=n_channels,
            ch_mults=channel_multiplers,
            is_attn=is_attention
        ).to(device=self.device)

        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device
        )

        self.dataset_name = dataset_name

        if eval_dataset is None:
            dataset_size = len(train_dataset)
            train_size = int(dataset_size * 0.9)
            eval_size = dataset_size - train_size
            train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])

        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True, pin_memory=True)
        self.eval_dataset = eval_dataset
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size, pin_memory=True)
        self.optimizer = Adam(self.eps_model.parameters(), lr=learning_rate)

        self.save_path = os.path.join(os.getcwd(), 'save')
        model_save_path = os.path.join(self.save_path, 'checkpoints', self.dataset_name + '_' + 'diffusion.pt')
        if os.path.exists(model_save_path):
            checkpoint = torch.load(model_save_path, weights_only=True)
            self.eps_model.load_state_dict(checkpoint['model_state_dict'])

    def sample(self, image_name='samples.png'):
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                                          device=self.device)
            
            with tqdm(range(self.n_steps)) as tepoch:
                tepoch.set_description("Sampling...")
                for t_ in tepoch:
                    t = self.n_steps - t_ - 1
                    x = self.diffusion.p_sample(x, x.new_full((self.n_samples,),  t, dtype=torch.int64))
            
        image_grid = make_grid(x, nrow=8, normalize=True)
        save_image(image_grid, os.path.join(self.save_path, 'images', image_name))

    def train(self, n_epochs):
        best_eval_loss = math.inf

        model_save_path = os.path.join(self.save_path, 'checkpoints', self.dataset_name + '_' + 'diffusion.pt')

        if os.path.exists(model_save_path):
            checkpoint = torch.load(model_save_path, weights_only=True)
            self.eps_model.load_state_dict(checkpoint['model_state_dict'])
            best_eval_loss = checkpoint['best_eval_loss']

        for epoch in range(n_epochs):
            # train eps model
            self.eps_model.train()
            with tqdm(self.train_dataloader) as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")

                for x in tepoch:
                    x = x.to(self.device)
                    self.optimizer.zero_grad()

                    loss = self.diffusion.loss(x)
                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(loss = loss.item())
            
            # evaluate eps model
            self.eps_model.eval()
            eval_loss = 0
            with torch.no_grad():
                for x in self.eval_dataloader:
                    x = x.to(self.device)
                    loss = self.diffusion.loss(x)
                    eval_loss += loss.item()

            print(f"eval loss: {eval_loss}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save({
                    'model_state_dict': self.eps_model.state_dict(),
                    'best_eval_loss': best_eval_loss,
                }, model_save_path)

            # sample
            if (epoch + 1) % 10 == 0:
                self.sample(image_name=f"Epoch_{epoch+1}_samples.png")



        

            