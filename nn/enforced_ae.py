import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.modules.cnn import ConvEncoder, ConvDecoder
from nn.modules.mlp import MLPEncoder, MLPBottleneck, MLPDecoder, MLP
import torchvision.transforms.v2.functional as F_v2


class EnforcedAE(nn.Module):
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

    def transform_images(self, images, action=None):
        # Sample action
        if action is None:
            action = torch.randn(5, dtype=images.dtype, device=images.device) * 0.4

        # Calculate affine parameters
        angle = action[0].item() * 180
        translate_x, translate_y = action[1].item() * 8, action[2].item() * 8
        scale = max(action[3].item() * 0.25 + 1.0, 0.1)
        shear = action[4].item() * 25

        # Apply affine transformation
        images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
        actions = action.unsqueeze(0).repeat(images.shape[0], 1)
        return images_aug, actions
    
    def interact(self, images, groups=8):
        """
        Interact with the images by applying either image or spectrogram transformations.
        
        Parameters:
        images (torch.Tensor): The input image tensor.
        groups (int): The number of groups to split the images into.
        
        Returns:
        torch.Tensor: The augmented images tensor.
        torch.Tensor: The actions tensor.

        """
        N, _, original_height, original_width = images.size()
        if N < groups:
            groups = N
        n_per = N // groups

        images_aug_arr = []
        actions_arr = []

        lo, hi = 0, n_per + N % groups
        while lo < N:
            images_aug, actions = self.transform_images(images[lo:hi])
            
            images_aug_arr.append(images_aug)
            actions_arr.append(actions)

            lo = hi
            hi = min(N, lo + n_per)
        
        return torch.cat(images_aug_arr, dim=0), torch.cat(actions_arr, dim=0)

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        z = self.infer(x)
        z[:,:5] += actions
        x_hat = self.decode(z)
        return x_hat, z

    def recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor):
        # x: (batch_size, 1, 28, 28)
        # x_hat: (batch_size, 1, 28, 28)
        return F.mse_loss(x_hat, x, reduction='none').sum(dim=[1, 2, 3])
    
    def loss(self, batch, **_):
        x, _ = batch
        x_aug, actions = self.interact(x)
        x_hat, _ = self.forward(x, actions)
        recon_loss = self.recon_loss(x_hat, x_aug).mean()
        return {'loss': recon_loss}