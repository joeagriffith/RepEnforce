import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.modules.cnn import ConvEncoder, ConvDecoder
from nn.modules.mlp import MLPEncoder, MLPBottleneck, MLPDecoder

class AE(nn.Module):
    def __init__(self, z_dim:int, cnn:bool=False, **kwargs):
        super().__init__()
        assert z_dim > 0, "z_dim must be greater than 0"
        self.z_dim = z_dim
        encoder = ConvEncoder(out_dim=512) if cnn else MLPEncoder(out_dim=512)
        bottleneck = MLPBottleneck(in_dim=512, out_dim=z_dim)
        self.encoder = nn.Sequential(encoder, bottleneck)
        self.decoder = ConvDecoder(in_dim=z_dim) if cnn else MLPDecoder(in_dim=z_dim)
    
    def rsample(self):
        raise NotImplementedError("This method is not used by AE")
    
    # Infer the latent variables
    def infer(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        return self.encoder(x)
    
    # Decode the latent variables, p(x|z)
    def decode(self, z: torch.Tensor):
        # z: (batch_size, z_dim)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        z = self.infer(x)
        x_hat = self.decode(z)
        return x_hat, z

    def recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor):
        return F.mse_loss(x_hat, x, reduction='none').sum(dim=[1, 2, 3])
    
    def loss(self, batch, **_):
        x, _ = batch
        x_hat, _ = self.forward(x)
        recon_loss = self.recon_loss(x_hat, x).mean()
        return {'loss': recon_loss}
    
    def embed(self, x: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        return self.encoder(x)
    
    def reconstruct(self, x: torch.Tensor):
        return self.forward(x)[0]