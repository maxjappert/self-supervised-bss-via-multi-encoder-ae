import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils import weight_norm

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
    def __init__(self, image_hw, channels, hidden):
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
                 norm_type, num_encoders, num_channels, image_hw,
                 kernel_size=7, stride=1, padding=3,):
        super(DecoderBlock, self).__init__()

        in_channels = in_channels# // num_encoders
        out_channels = out_channels# // num_encoders

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
            self.block.append(nn.GroupNorm(num_encoders, out_channels))
        elif norm_type == 'layer_norm':
            down_sample = (num_channels-2) - c_i
            self.block.append(nn.LayerNorm([out_channels, int(image_hw//(2**down_sample)), int(image_hw//(2**down_sample))]))
        elif norm_type == 'instance_norm':
            self.block.append(nn.InstanceNorm2d(out_channels))
        
    def forward(self, x):
        return self.block(x)
    
class ConvolutionalDecoder(nn.Module):
    def __init__(self, image_hw, channels, hidden, num_encoders, norm_type='none'):
        super(ConvolutionalDecoder, self).__init__()
        self.image_hw = image_hw
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
            self.decoder.append(nn.GroupNorm(num_encoders, channels[-1]))
        elif norm_type == 'layer_norm':
            down_sample = (channels[-1]-2) - 0
            self.decoder.append(nn.LayerNorm([channels[-1], image_hw//(2**down_sample), image_hw//(2**down_sample)]))
        elif norm_type == 'instance_norm':
            self.decoder.append(nn.InstanceNorm2d(channels[-1]))
        for c_i in reversed(range(1, len(channels))):
            self.decoder.append(DecoderBlock(channels[c_i], channels[c_i-1], c_i, 
                                             norm_type, num_encoders, len(channels), 
                                             image_hw))
        
    def forward(self, z):
        #z = torch.concatenate(z, dim=1)
        y = self.decoder(z)

        return y
            

class LinearConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels=3, image_height=64, image_width=64,
                 channels=[32, 64, 128], hidden=512, 
                 num_encoders=4, norm_type='none',
                 use_weight_norm=True, return_sum=True):
        super(LinearConvolutionalAutoencoder, self).__init__()
        self.image_hw = image_hw
        self.input_channels = input_channels
        self.channels = channels
        self.hidden = hidden
        self.num_encoders = num_encoders
        self.return_sum = return_sum
        
        # encoder layers
        enc_channels = [c//num_encoders for c in channels]
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(num_encoders):
            self.encoders.append(ConvolutionalEncoder(image_hw=image_hw,
                                                      channels=[input_channels] + enc_channels,
                                                      hidden=hidden//num_encoders))

        for _ in range(num_encoders):
            self.decoders.append(ConvolutionalDecoder(image_hw=image_hw,
                                            channels=[channels[0]] + channels,
                                            hidden=hidden//num_encoders, num_encoders=num_encoders,
                                            norm_type=norm_type))


        # output layer
        self.outputs = nn.ModuleList()
        for i in range(num_encoders):
            if use_weight_norm:
                self.outputs.append(nn.Sequential(
                    weight_norm(nn.Conv2d(in_channels=channels[0],
                                          out_channels=input_channels,
                                          kernel_size=1, stride=1,
                                          padding=0))
                ))
            else:
                self.outputs.append(nn.Sequential(
                    nn.Conv2d(in_channels=channels[0],
                              out_channels=input_channels,
                              kernel_size=1, stride=1,
                              padding=0)
                ))
    
    def encode(self, x):
        z = []
        for i, encoder in enumerate(self.encoders):
            e = encoder(x)
            z.append(e)
            
        return z
    
    def decode(self, z, zeros_train=False):
        if zeros_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)\
                    or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    m.eval()

        ys = []
        for i, decoder in enumerate(self.decoders):
            ys.append(self.outputs[i](decoder(z[i])))
        
        if zeros_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)\
                    or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                    m.train()

        return sum(ys) if self.return_sum else ys
    
    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        
        return y, z
