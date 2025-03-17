import torch
import torch.nn as nn

class UpBlockUnet(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 num_heads, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ),
            ]
        )
        
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
        ])
            
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(norm_channels, out_channels),
                nn.GroupNorm(norm_channels, out_channels)
            ]
        )
        
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True),
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            ]
        )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1)    
        
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        out = x
        
        # Resnet1
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        # Self Attention
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norms[0](in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attentions[0](in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        # Resnet2
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        
        # Self Attention
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norms[1](in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attentions[1](in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        return out

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=3, t_emb_dim=256, num_heads=4, norm_channels=32):
        super().__init__()
        
        # Time embedding
        self.t_embedder = nn.Sequential(
            nn.Linear(1, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Upsampling path
        self.up1 = UpBlockUnet(512, 128, t_emb_dim, num_heads, 2, norm_channels)
        self.up2 = UpBlockUnet(256, 64, t_emb_dim, num_heads, 2, norm_channels)
        
        # Output convolution
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.t_embedder(t.unsqueeze(-1).float())
        
        # Initial features
        x1 = self.conv_in(x)
        
        # Downsampling
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Upsampling
        x = self.up1(x3, x2, t_emb)
        x = self.up2(x, x1, t_emb)
        
        # Output
        out = self.conv_out(x)
        
        return out
