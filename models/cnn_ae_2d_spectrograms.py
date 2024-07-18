import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchvision import transforms


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, momentum=0.8),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels, momentum=0.8),
        )
        
    def forward(self, x):
        return self.block(x)

class ConvolutionalEncoder(nn.Module):
    def __init__(self, image_h, image_w, channels, hidden):
        super(ConvolutionalEncoder, self).__init__()
        
        self.encoder = nn.Sequential()
        for c_i in range(len(channels)-1):
            self.encoder.append(EncoderBlock(channels[c_i], channels[c_i+1]))
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels[-1], hidden, kernel_size=1,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        z = self.encoder_conv(x)
        
        return z
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, c_i, 
                 norm_type, num_channels, image_h, image_w,
                 kernel_size=7, stride=1, padding=3,):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.block.append(nn.ConvTranspose2d(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, stride=stride, 
                                             padding=padding))
        self.block.append(nn.ReLU(inplace=True))
        if norm_type == 'batch_norm':
            self.block.append(nn.BatchNorm2d(out_channels, momentum=0.8))
        elif norm_type == 'group_norm':
            self.block.append(nn.GroupNorm(1, out_channels))
        elif norm_type == 'layer_norm':
            down_sample = (num_channels-2) - c_i
            self.block.append(nn.LayerNorm([out_channels, int(image_h//(2**down_sample)), int(image_w//(2**down_sample))]))
        elif norm_type == 'instance_norm':
            self.block.append(nn.InstanceNorm2d(out_channels))
        
    def forward(self, x):
        return self.block(x)
    
class ConvolutionalDecoder(nn.Module):
    def __init__(self, image_h, image_w, channels, hidden, kernel_size=7, norm_type='none'):
        super(ConvolutionalDecoder, self).__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.num_channels = len(channels)
        self.channels = channels
        
        NORM_TYPES = ['none', 'batch_norm', 'group_norm', 'layer_norm', 'instance_norm']
        assert norm_type in NORM_TYPES, f'Given norm type, {norm_type}, not in {NORM_TYPES}.'
        
        # create convolutional decoder
        self.decoder = nn.Sequential()
        self.decoder.append(nn.Conv2d(hidden, channels[-1], kernel_size=1, 
                            stride=1, padding=0))
        self.decoder.append(nn.ReLU(inplace=True))
        if norm_type == 'batch_norm':
            self.decoder.append(nn.BatchNorm2d(channels[-1], momentum=0.8))
        elif norm_type == 'group_norm':
            self.decoder.append(nn.GroupNorm(1, channels[-1]))
        elif norm_type == 'layer_norm':
            down_sample = (channels[-1]-2) - 0
            self.decoder.append(nn.LayerNorm([channels[-1], image_h//(2**down_sample), image_w//(2**down_sample)]))
        elif norm_type == 'instance_norm':
            self.decoder.append(nn.InstanceNorm2d(channels[-1]))
        for c_i in reversed(range(1, len(channels))):
            self.decoder.append(DecoderBlock(channels[c_i], channels[c_i-1], c_i, 
                                             norm_type, len(channels),
                                             image_h, image_w, kernel_size=kernel_size))
        
    def forward(self, z):
        y = self.decoder(z)

        print(y.shape)

        return transforms.Resize((self.image_h, self.image_w))(y)
            

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels=3, image_h=64, image_w=64,
                 channels=[32, 64, 128], hidden=512, 
                 norm_type='none',
                 use_weight_norm=True, kernel_size=7):
        super(ConvolutionalAutoencoder, self).__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.input_channels = input_channels
        self.channels = channels
        self.hidden = hidden

        # encoder layers
        enc_channels = channels
        self.encoder = ConvolutionalEncoder(image_h=image_h, image_w=image_w,
                                                      channels=[input_channels] + enc_channels,
                                                      hidden=hidden)

        # decoder layers
        self.decoder = ConvolutionalDecoder(image_h=image_h, image_w=image_w,
                                            channels=[channels[0]] + channels,
                                            hidden=hidden,
                                            norm_type=norm_type, kernel_size=kernel_size)
        # output layer
        if use_weight_norm:
            self.output = nn.Sequential(
                weight_norm(nn.Conv2d(in_channels=channels[0], 
                                      out_channels=input_channels, 
                                      kernel_size=1, stride=1,
                                      padding=0))
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=channels[0], 
                          out_channels=input_channels, 
                          kernel_size=1, stride=1, 
                          padding=0)
            )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z, zeros_train=False):
        if zeros_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)\
                    or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    m.eval()

        y = self.decoder(z)
        y = self.output(y)
        
        if zeros_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)\
                    or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                    m.train()
        
        return y
    
    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        
        return y, z