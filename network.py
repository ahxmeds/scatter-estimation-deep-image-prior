#%%
import torch 
import torch.nn as nn 
#%%
def get_downward_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(), 
        ) 
    
def get_downsample_block(out_channels):
    return nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        ) 
    
def get_bottleneck_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        ) 
    
def get_bilinear_upsample_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same'),
        )
    
def get_upward_block(in_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
        )
    
def get_final_block(in_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding='same'),
        )
#%%
class UNetCustom(nn.Module):
    def __init__(self, n_channels=[2, 4, 8, 16, 32]):
        super().__init__()       
        self.num_layers = len(n_channels) - 1
        
        # list of downward blocks
        self.first_block = get_downward_block(1, n_channels[0])
        self.downward_blocks = nn.ModuleList([get_downward_block(n_channels[i], n_channels[i+1]) for i in range(self.num_layers-1)])
        self.downward_blocks = nn.ModuleList([self.first_block] + list(self.downward_blocks))

        # list of downsample blocks
        self.downsample_blocks = nn.ModuleList([get_downsample_block(n_channels[i]) for i in range(self.num_layers)])

        # bottleneck blocks
        self.bottleneck_block = get_bottleneck_block(n_channels[self.num_layers-1], n_channels[self.num_layers])
        
        # list of unsample blocks
        self.upsample_blocks = nn.ModuleList([get_bilinear_upsample_block(n_channels[self.num_layers-i], n_channels[self.num_layers-i-1]) for i in range(self.num_layers)])
        
        # list of upward blocks
        self.upward_blocks = nn.ModuleList([get_upward_block(n_channels[self.num_layers-i-1]) for i in range(self.num_layers-1)])
        self.final_block = get_final_block(n_channels[0])
        self.upward_blocks = nn.ModuleList(list(self.upward_blocks) + [self.final_block])
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_outputs = []
        for i in range(self.num_layers):
            x = self.downward_blocks[i](x)
            down_outputs.append(x)
            x = self.downsample_blocks[i](x)

        x = self.bottleneck_block(x)

        for i in range(self.num_layers):
            x = self.upsample_blocks[i](x)
            x = x + down_outputs[self.num_layers - i - 1]  # additive skip-connections
            x = self.upward_blocks[i](x)

        return x
