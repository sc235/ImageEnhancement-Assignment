import torch
import torch.nn as nn
from typing import Optional, Union

class DnCNN(nn.Module):
    """
    DnCNN (Deep Convolutional Neural Network for Image Denoising)
    Paper: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
           Zhang et al., IEEE TIP 2017.
    
    Key Idea:
      Rather than predicting the clean image x directly, DnCNN uses residual learning
      to predict the noise map R(y) ~= v. The clean image is then recovered as:
          x_hat = y - R(y)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 64, num_layers: int = 20):
        super(DnCNN, self).__init__()
        layers = []
        
        # 1. First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # 2. Intermediate layers: Conv + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            
        # 3. Output layer: Conv (predicts 3-channel residual noise)
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Noisy image tensor of shape [B, C, H, W] in range [0, 1]
        Returns:
            Denoised image tensor of shape [B, C, H, W] in range [0, 1]
        """
        residual_noise = self.model(x)
        return torch.clamp(x - residual_noise, 0.0, 1.0)


class FFDNet(nn.Module):
    """
    FFDNet (Fast and Flexible Denoising Network)
    Paper: "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
           Zhang et al., IEEE TIP 2018.
    
    Key Innovations:
      1. Noise Level Map: Takes a tunable noise level sigma as an explicit input channel.
         Allows one single network to handle arbitrary uniform or spatially-varying noise.
      2. Reversible Sub-sampling (PixelUnshuffle): Downsamples (C, H, W) into (4C, H/2, W/2).
         Reduces computation by 4x, drastically accelerating inference speed while expanding receptive field.
      3. Sub-pixel Convolution (PixelShuffle): Reconstructs full-resolution clean output.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 96, num_layers: int = 12):
        super(FFDNet, self).__init__()
        self.in_nc = in_channels
        self.out_nc = out_channels
        self.scale_factor = 2
        
        # Total input channels: 4 sub-images * in_channels + 1 noise level map channel
        head_in_channels = in_channels * (self.scale_factor ** 2) + 1
        
        layers = []
        # First layer
        layers.append(nn.Conv2d(head_in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            
        # Final output layer
        layers.append(nn.Conv2d(num_features, out_channels * (self.scale_factor ** 2), kernel_size=3, padding=1, bias=True))
        
        self.model = nn.Sequential(*layers)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.scale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Noisy image tensor [B, C, H, W] in [0, 1]
            sigma: Scalar float (e.g. 25.0 / 255.0) or noise map tensor [B, 1, H/2, W/2]
        Returns:
            Denoised image tensor [B, C, H, W] in [0, 1]
        """
        h, w = x.size()[-2:]
        pad_h = int((2 - h % 2) % 2)
        pad_w = int((2 - w % 2) % 2)
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        hp, wp = x.size()[-2:]
        
        # 1. Reversible sub-sampling
        x_down = self.pixel_unshuffle(x)
        
        # 2. Formulate noise level map
        if not torch.is_tensor(sigma):
            sigma_map = torch.full((x.size(0), 1, hp // 2, wp // 2), float(sigma), device=x.device, dtype=x.dtype)
        else:
            if sigma.dim() == 2:
                sigma_map = sigma.unsqueeze(0).unsqueeze(0)
            elif sigma.dim() == 3:
                sigma_map = sigma.unsqueeze(1)
            else:
                sigma_map = sigma
            if sigma_map.shape[-2:] != (hp // 2, wp // 2):
                sigma_map = nn.functional.interpolate(sigma_map, size=(hp // 2, wp // 2), mode='bilinear', align_corners=False)
            
        # 3. Concatenate downsampled features with noise level map
        x_in = torch.cat([x_down, sigma_map], dim=1)
        
        # 4. Convolutional backbone
        feat = self.model(x_in)
        
        # 5. Sub-pixel upsampling
        out = self.pixel_shuffle(feat)
        
        # Crop back to original dimensions if padded
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
            
        return torch.clamp(out, 0.0, 1.0)


def load_dncnn(checkpoint_path: str, device: str = "cpu") -> DnCNN:
    """Loads pre-trained DnCNN model from checkpoint."""
    model = DnCNN(in_channels=3, out_channels=3, num_features=64, num_layers=20)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_ffdnet(checkpoint_path: str, device: str = "cpu") -> FFDNet:
    """Loads pre-trained FFDNet model from checkpoint."""
    model = FFDNet(in_channels=3, out_channels=3, num_features=96, num_layers=12)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
