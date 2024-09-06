import torch
import torch.nn as nn
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
import lpips

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set the desired seed value #
seed = 42
# This is for accumulating gradients incase the images are huge
# And one cant afford higher batch sizes
acc_steps = 1
image_save_steps = 5
img_save_count = 0
num_epochs = 10
lr = 0.0001
disc_step_start = 1000
step_count = 0
batch_size = 64
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


class CustomDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.image_names = os.listdir(img_folder) # Get the list of image names in the folder

    def __getitem__(self, index):
        # Get image name from the list
        img_name = self.image_names[index]

        # Open image
        image = Image.open(os.path.join(self.img_folder, img_name))

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_names)

transform = transforms.Compose(
    [
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
       transforms.Normalize(
            [0.5 for _ in range(3)], [0.5 for _ in range(3)]),
            
            ]
)
# Example dataset instantiation
dataset = CustomDataset(img_folder='/csehome/m23mac008/dl5/isic2016/train', transform=transform)

# Example dataloader creation
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)



# architecture of the model

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
                    nn.GroupNorm(norm_channels, in_channels ),
                    nn.SiLU(),
                    nn.Conv2d(in_channels , out_channels, kernel_size=3, stride=1,
                              padding=1),),

                nn.Sequential(nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d( out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1)),
                
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d( out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1)),
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
            [nn.GroupNorm(norm_channels, out_channels),
             nn.GroupNorm(norm_channels, out_channels),]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True),
             nn.MultiheadAttention(out_channels, num_heads, batch_first=True),]
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
        
        # First input  block
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
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                ),
                nn.Sequential(
                    nn.GroupNorm(norm_channels,out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
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
                nn.Conv2d(in_channels , out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels,4, 2, 1)  
          
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
    def __init__(self):
        super().__init__()
        
    
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_layers = nn.Sequential(
            DownBlock(64, 128, 32),
            DownBlock(128, 256, 32),
            DownBlock(256, 256, 32)
        )

        self.encoder_mids = nn.Sequential(
            MidBlock(256, 256, 4, 32)
        )

        self.encoder_norm_out = nn.GroupNorm(32, 256)
        self.encoder_conv_out = nn.Conv2d(256, 3, kernel_size=3, padding=1)

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(3, 3, kernel_size=1)

        # Codebook
        self.embedding = nn.Embedding(8192, 3)
        
        
        ##################### Decoder ######################

        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(3, 3, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(3, 256, kernel_size=3, padding=(1, 1))

        # Midblock + Upblock
        self.decoder_mids = nn.Sequential(
            MidBlock(256, 256, 4, 32)
        )

        self.decoder_layers = nn.Sequential(
            UpBlock(256, 256, 32),
            UpBlock(256, 128, 32),
            UpBlock(128, 64, 32)
        )

        self.decoder_norm_out = nn.GroupNorm(32, 64)
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
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
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
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses
    
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
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses
    


#discriminator model

class Discriminator(nn.Module):
    
    def __init__(self, im_channels=3):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.im_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            activation
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            activation
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Identity()  # No activation or batch normalization for the last layer
        )

    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out



# initiate a model
model = VQVAE().to(device)

discriminator = Discriminator(im_channels=3).to(device)



recon_criterion = torch.nn.MSELoss()
disc_criterion = torch.nn.MSELoss()
lpips_model = lpips.LPIPS(net='vgg').to(device)

optimizer_d = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_g = Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch_idx in range(num_epochs):
    recon_losses = []
    codebook_losses = []
    perceptual_losses = []
    disc_losses = []
    gen_losses = []
    losses = []
    
    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    
    for image in data_loader:
        step_count += 1
        image = image.float().to(device)
        
        # Fetch autoencoders output(reconstructions)
        model_output = model(image)
        output, z, quantize_losses = model_output
        
        ######### Optimize VQVAEGAN ##########
        # L2 Loss
        recon_loss = recon_criterion(output, image) 
        recon_losses.append(recon_loss.item())
        recon_loss = recon_loss / acc_steps
        g_loss = (recon_loss +
                (1 * quantize_losses['codebook_loss'] / acc_steps)+
                (0.2 * quantize_losses['commitment_loss'] / acc_steps))
        
        codebook_losses.append(1* quantize_losses['codebook_loss'].item())

        # Adversarial loss only if disc_step_start steps passed
        if step_count > disc_step_start:
            disc_fake_pred = discriminator(model_output[0])
            disc_fake_loss = disc_criterion(disc_fake_pred, torch.ones(disc_fake_pred.shape,
                                                        device=disc_fake_pred.device))
            
            gen_losses.append(0.5 * disc_fake_loss.item())
            g_loss += 0.5 * disc_fake_loss / acc_steps

        lpips_loss = lpips_model(image,output) # LPIPS Loss
        perceptual_losses.append(1 * lpips_loss.item())
        g_loss += 1*lpips_loss / acc_steps
        losses.append(g_loss.item())
        g_loss.backward()
        
        ######### Optimize Discriminator ##########
        if step_count > disc_step_start:
            fake = output
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(image)
            disc_fake_loss = disc_criterion(disc_fake_pred,
                                            torch.zeros(disc_fake_pred.shape,
                                                        device=disc_fake_pred.device))
            disc_real_loss = disc_criterion(disc_real_pred,
                                            torch.ones(disc_real_pred.shape,
                                                        device=disc_real_pred.device))
            disc_loss = 0.5 * (disc_fake_loss + disc_real_loss) / 2

            disc_losses.append(disc_loss.item())
            disc_loss = disc_loss / acc_steps
            disc_loss.backward()
            if step_count % acc_steps == 0:
                optimizer_d.step()
                optimizer_d.zero_grad()
        
        if step_count % acc_steps == 0:
            optimizer_g.step()
            optimizer_g.zero_grad()

    optimizer_d.step()
    optimizer_d.zero_grad()
    optimizer_g.step()
    optimizer_g.zero_grad()

    sample_size = min(8, image.shape[0])
    save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
    save_output = ((save_output + 1) / 2)
    save_input = ((image[:sample_size] + 1) / 2).detach().cpu()
        
    grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(os.path.join('result','current_autoencoder_sample_{}.png'.format(img_save_count)))
    img_save_count += 1
    img.close()

    # save weights with epoch no in name 
    torch.save(model.state_dict(), os.path.join('result','model_epoch_{}.pth'.format(epoch_idx)))
        
    print(
        'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
        'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
        format(epoch_idx + 1,
                np.mean(recon_losses),
                np.mean(perceptual_losses),
                np.mean(codebook_losses),
                np.mean(gen_losses),
                np.mean(disc_losses)))

  
print('Done Training...')

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Calculate betas linearly
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Calculate alphas
        self.alphas = 1.0 - (self.betas - beta_start) / (beta_end - beta_start)

        # Compute cumulative product of alphas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

        # Calculate square roots
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        # Reshape tensors
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size, 1, 1, 1)

        # Apply forward process equation
        return (sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        # Calculate mean and variance
        mean = xt - ((self.betas[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t])
        mean /= torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, ((xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) /
                          torch.sqrt(self.alpha_cum_prod[t]))
        else:
            variance = (1 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t]) * self.betas[t]
            sigma = torch.sqrt(variance)
            z = torch.randn_like(xt)
            return mean + sigma * z, mean / torch.sqrt(1 - self.alpha_cum_prod[t - 1])

class UpBlockUnet(nn.Module):
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 num_heads, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        # print weight size of nn.GroupNorm(norm_channels, in_channels),
        print(norm_channels,in_channels)
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels , out_channels, kernel_size=3, stride=1,
                              padding=1),
                ),
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
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
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1)    
        
    def forward(self, x, out_down,t_emb):
 
        #print('x',x.shape)
        x = self.up_sample_conv(x)
        #print('x',x.shape)
        #print('out_down.shape',out_down.shape)
        x = torch.cat([x, out_down], dim=1)
        #print('x',x.shape)
        out = x
        # Resnet1
        resnet_input = out
        print(out.shape)
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
    
    
##unet






# do a forward pass to check the output shape of the model
x = torch.randn(2, im_channels,4,4).to(device)
t = torch.randint(0, 1000, (2,)).to(device)
out = model(x, t)
print(out.shape)

#load save weights
#vqvae.load_state_dict(torch.load('vqvae_ckpt.pth'))

num_epochs = 100
optimizer = Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

# Create the noise scheduler
scheduler = LinearNoiseScheduler(num_timesteps=1000,
                                    beta_start=0.0015,
                                    beta_end=0.0195)
    
#freze vqvae
for param in vqvae.parameters():
    param.requires_grad = False

for epoch_idx in range(num_epochs):
    losses = []
    for im in train_loader:
        optimizer.zero_grad()
        im = im.float().to(device)
        with torch.no_grad():                           
                im, _ = vqvae.encode(im)
        
        # Sample random noise
        noise = torch.randn_like(im).to(device)
        
        # Sample timestep
        t = torch.randint(0, 1000, (im.shape[0],)).to(device)
        
        # Add noise to images according to timestep
        noisy_im = scheduler.add_noise(im, noise, t)
        noise_pred = model(noisy_im, t)
        
        loss = criterion(noise_pred, noise)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch:{} | Loss : {:.4f}'.format(
        epoch_idx + 1,
        np.mean(losses)))

print('Done Training ...')


##infer

with torch.no_grad():
    im_size = 32 // 2**3
    xt = torch.randn((1, 3, im_size, im_size)).to(device)

    # Get prediction of noise at the final timestep
    noise_pred = model(xt, torch.tensor(0).unsqueeze(0).to(device))

    # Use scheduler to get x0 prediction
    xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.tensor(0).to(device))

    # Decode the final image
    ims = vqvae.decode(xt)
    ims = torch.clamp(ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2

    grid = make_grid(ims, nrow=1)
    img = torchvision.transforms.ToPILImage()(grid)

    img.save(os.path.join('samples_dir', 'x1_final.png'))
    img.close()

with torch.no_grad():
    num_samples = 100  # 8 columns, 2 rows
    xt = torch.randn((num_samples,3,4,4)).to(device)


    # Get prediction of noise at the final timestep
    noise_pred = model(xt, torch.tensor(0).unsqueeze(0).to(device))

    # Use scheduler to get x0 prediction
    xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.tensor(0).to(device))

    # Decode the final image
    ims = vqvae.decode(xt)
    ims = torch.clamp(ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2

    grid = make_grid(ims, nrow=10)  # 8 columns
    img = torchvision.transforms.ToPILImage()(grid)

    img.save(os.path.join('samples_dir', 'x0_final_grid_1.png'))
    img.close()

    