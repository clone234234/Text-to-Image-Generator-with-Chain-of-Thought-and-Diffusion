import torch 
from torch import nn
from torch.nn import functional as F
from attention import self_attention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.attention = self_attention(in_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue=x
        n,c,h, w = x.shape
        x= x.view(n, c, h * w)
        x=x.transpose(-1, -2)
        x = self.attention(x)
        x= x.view((n, c, h , w))
        x+= residue
        return x

        
        

        
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1= nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue=x
        x= self.groupnorm1(x)
        x= F.silu(x)
        x= self.conv1(x)
        x= self.groupnorm2(x)
        x= F.silu(x)
        x= self.conv2(x)
        return x + self.residual_layer(residue)
    

class encoder(nn.Sequential):
    def __init__(self):
        super().__init__ (
            nn.Conv2d(3,128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256,256),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3,  padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0))


        
    def forward(self, x: torch.Tensor, noise: torch.tensor) -> torch.Tensor:
        for model in self:
            if getattr(model, 'tride', None) ==(2,2):
                x= F.pad(x, (0, 0, 1, 1))
            x = model(x)
        mean, log_Variance = torch.chunk(x, 2, dim=1)  
        log_Variance = torch.clamp(log_Variance, -30.0, 20.0)
        variance= log_Variance.exp()
        stdev= variance.sqrt()
        x = mean + stdev * noise
        x*= 0.18215
        return x, mean, log_Variance    
    
class decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128,4, kernel_size=3, padding=1),

       
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215
        for module in self:
            x = module(x)
        return x