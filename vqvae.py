import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_channels):
        super().__init__()
        self.resnet_conv_first = nn.Sequential(
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
        )
        self.resnet_conv_second = nn.Sequential(
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ),
        )
        self.residual_input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        out = x

        # first Resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # Second Resnet block
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        # Downsample
        out = self.down_sample_conv(out)

        return out
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, norm_channels):
        super().__init__()
        self.nam_heads = num_heads
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
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ),
            ]
        )

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
                ),
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ),
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(norm_channels, out_channels),
                nn.GroupNorm(norm_channels, out_channels),
            ]
        )
        
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True),
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True),
            ]
        )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
            ]
        )
    
    def forward(self, x):
        out = x
        
        # First input block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
    
        # Attention Block 1
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norms[0](in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attentions[0](in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        # Resnet Block 1
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        
        # Attention Block 2
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norms[1](in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attentions[1](in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
            
        # Resnet Block 2 
        resnet_input = out
        out = self.resnet_conv_first[2](out)
        out = self.resnet_conv_second[2](out)
        out = out + self.residual_input_conv[2](resnet_input)
        
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_channels):
        super().__init__()
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
                )
            ]
        )
        
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

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)  
          
    def forward(self, x):
        # Upsample
        x = self.up_sample_conv(x)
        
        out = x

        # Resnet Block 1
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        # Resnet Block 2
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        return out

class VQVAE(nn.Module):
    def __init__(self, codebook_size=8192, embedding_dim=3, norm_channels=32, num_heads=4):
        super().__init__()
        
        # Encoder
        self.encoder_conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Downblock + Midblock
        self.encoder_layers = nn.Sequential(
            DownBlock(64, 128, norm_channels),
            DownBlock(128, 256, norm_channels),
            DownBlock(256, 256, norm_channels)
        )

        self.encoder_mids = nn.Sequential(
            MidBlock(256, 256, num_heads, norm_channels)
        )

        self.encoder_norm_out = nn.GroupNorm(norm_channels, 256)
        self.encoder_conv_out = nn.Conv2d(256, embedding_dim, kernel_size=3, padding=1)

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)

        # Codebook
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        
        # Decoder
        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(embedding_dim, 256, kernel_size=3, padding=1)

        # Midblock + Upblock
        self.decoder_mids = nn.Sequential(
            MidBlock(256, 256, num_heads, norm_channels)
        )

        self.decoder_layers = nn.Sequential(
            UpBlock(256, 256, norm_channels),
            UpBlock(256, 128, norm_channels),
            UpBlock(128, 64, norm_channels)
        )

        self.decoder_norm_out = nn.GroupNorm(norm_channels, 64)
        self.decoder_conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def quantize(self, x):
        B, C, H, W = x.shape
        
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        
        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        
        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()
        
        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)
        out = self.encoder_layers(out)
        out = self.encoder_mids(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, indices = self.quantize(out)
        return out, quant_losses, indices
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        out = self.decoder_mids(out)
        out = self.decoder_layers(out)
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, quant_losses, _ = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses