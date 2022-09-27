import pytorch_lightning as pl
from torch import nn, optim

class CAE(pl.LightningModule):
	def __init__(self, latent_size=10):
		super().__init__()
		
		self.conv_encoder = nn.Sequential(
			nn.Conv2D(1, 32, 2, stride=2),
			nn.ReLU(),
			nn.Conv2D(32, 64, 2, stride=2),
			nn.ReLU(),
			nn.Conv2D(64, 128, 2, stride=2),
			nn.ReLU(),
			nn.Flatten()
		)
		self.latent = nn.Linear(1152, latent_size),
		self.conv_decoder = nn.Sequential(
			# TODO
		)
