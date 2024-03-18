#%%
from __future__ import annotations
import os
from pytomography.io.SPECT import simind
import matplotlib.pyplot as plt
import pytomography
import torch
import numpy as np
import torch

import pytomography
from torch.optim import LBFGS
import torch
import torch.nn as nn
from network import UNetCustom
import pandas as pd 
torch.set_default_device('cuda:0')
pytomography.device='cuda:0'

#%%
def convert_to_N_digits(num, N):
    return  str(num).zfill(N)


# %%
path = '/home/sahamed/Projects/pytomography/scatter-estimation-deep-image-prior/simind_tutorial'
organs = ['bkg', 'liver', 'l_lung', 'r_lung', 'l_kidney', 'r_kidney','salivary', 'bladder']
activities = [2500, 450, 7, 7, 100, 100, 20, 90] # MBq
headerfiles_photopeak = [os.path.join(path, 'multi_projections', organ, 'photopeak.h00') for organ in organs]
#%%
# noisy scatter
dT = 15 # seconds per projection
headerfiles_lower = [os.path.join(path, 'multi_projections', organ, 'lowerscatter.h00') for organ in organs]
headerfiles_upper = [os.path.join(path, 'multi_projections', organ, 'upperscatter.h00') for organ in organs]
projections = simind.get_projections([headerfiles_photopeak, headerfiles_lower, headerfiles_upper], weights=activities)
projections_realization = torch.poisson(projections * dT)
ww_peak = simind.get_energy_window_width(headerfiles_photopeak[0])
ww_lower = simind.get_energy_window_width(headerfiles_lower[0])
ww_upper = simind.get_energy_window_width(headerfiles_upper[0])
lower_scatter = projections_realization[1].unsqueeze(0)
upper_scatter = projections_realization[2].unsqueeze(0)
scatter = (lower_scatter/ww_lower+upper_scatter/ww_upper)*ww_peak/2
#%% 
# scatter noiseless (ground truth)
lower_scatter_noiseless = projections[1].unsqueeze(0) * dT
upper_scatter_noiseless = projections[2].unsqueeze(0) * dT
scatter_noiseless = (lower_scatter_noiseless/ww_lower+upper_scatter_noiseless/ww_upper)*ww_peak/2
#############################################################################################
# %%

# #%%
exp_code = 'unet_2x4x8'
num_channels = [int(i) for i in exp_code.split('_')[1].split('x')]
print(num_channels)
unet = UNetCustom(n_channels=num_channels)
truth = scatter[0].unsqueeze(1)
torch.manual_seed(42)
input = torch.randn(truth.shape)
torch.save(input, 'random_noise_input.pt')
optimizer_lfbgs = LBFGS(unet.parameters(), lr=1, max_iter=20, history_size=100)
criterion = torch.nn.MSELoss()
unet.train()
n_epochs = 10

#%%
save_dir = '/home/sahamed/Projects/pytomography/scatter-estimation-deep-image-prior/network_ablation_experiments'
logs_dir = os.path.join(save_dir, 'logs', f'{exp_code}')
models_dir = os.path.join(save_dir, 'models', f'{exp_code}')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
logs_fpath = os.path.join(logs_dir, 'logs.csv')
loss_list = []

def get_closure(optimizer):
    optimizer.zero_grad()
    NM_prediction = unet(input)
    loss = criterion(NM_prediction, truth)
    loss.backward()
    return loss


for epoch in range(n_epochs):  
    loss = optimizer_lfbgs.step(lambda: get_closure(optimizer_lfbgs))
    loss_list.append(loss.item())
    model_save_fpath = os.path.join(models_dir, f'checkpoint={convert_to_N_digits(epoch+1, 2)}.pth')
    torch.save(unet.state_dict(), model_save_fpath)
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


df = pd.DataFrame(loss_list, columns=['Loss'])
df.to_csv(logs_fpath, index=False)



# # %%
# exp_code = 'unet_32x64x128x256x512x1024'
# save_dir = '/home/sahamed/Projects/pytomography/scatter-estimation-deep-image-prior/network_ablation_experiments'
# models_dir = os.path.join(save_dir, 'models', f'{exp_code}')
# preds_dir = os.path.join(save_dir, 'predictions', f'{exp_code}')
# os.makedirs(preds_dir, exist_ok=True)
# model_fpath = os.path.join(models_dir, 'checkpoint=10.pth')
# pred_fpath = os.path.join(preds_dir, 'pred.pt')

# #%%
# num_channels = [int(i) for i in exp_code.split('_')[1].split('x')]
# print(num_channels)
# unet = UNetCustom(n_channels=num_channels)
# unet.load_state_dict(torch.load(model_fpath))
# unet.eval()
# truth = scatter[0].unsqueeze(1)
# torch.manual_seed(42)
# input = torch.randn(truth.shape)

# scatter_pred = unet(input)
# scatter_pred = scatter_pred.swapaxes(1,0)
# torch.save(scatter_pred, pred_fpath)
# # %%
