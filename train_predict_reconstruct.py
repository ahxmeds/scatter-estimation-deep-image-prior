#%%
from __future__ import annotations
import os
from pytomography.io.SPECT import simind
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
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
import time 

#%%
def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)


# exp_code = 'unet_2x4x8x16'
# exp_code = 'unet_8x16x32x64'
# exp_code = 'unet_8x16x32x64x64'
# exp_code = 'unet_8x16x32x64x128'
# exp_code = 'unet_16x32x64x128x256'
# exp_code = 'unet_16x32x64x128x256x512'
exp_codes = [
    'unet_8x16x32x64x64',
    'unet_8x16x32x64x128',
    'unet_16x32x64x128x256',
    'unet_16x32x64x128x256x512'
]

path = '/home/sahamed/Projects/pytomography/scatter-estimation-deep-image-prior/simind_tutorial'
organs = ['bkg', 'liver', 'l_lung', 'r_lung', 'l_kidney', 'r_kidney','salivary', 'bladder']
activities = [2500, 450, 7, 7, 100, 100, 20, 90] # MBq
headerfiles_photopeak = [os.path.join(path, 'multi_projections', organ, 'photopeak.h00') for organ in organs]
headerfiles_lower = [os.path.join(path, 'multi_projections', organ, 'lowerscatter.h00') for organ in organs]
headerfiles_upper = [os.path.join(path, 'multi_projections', organ, 'upperscatter.h00') for organ in organs]

#############################################################################################
# %%

# #%%
start_time = time.time()
dT_list = [0.015, 0.03, 0.1, 0.33, 1, 2, 5, 15]
for exp_code in exp_codes:
    for dT in dT_list:
        print(f'Projection time: {dT}')
        save_dir = f'/home/sahamed/Projects/pytomography/scatter-estimation-deep-image-prior/ablation_experiments_mean/dT{dT}_NoiseInput'
        for run in range(20):
            projections = simind.get_projections([headerfiles_photopeak, headerfiles_lower, headerfiles_upper], weights=activities)
            projections_realization = torch.poisson(projections * dT)
            ww_peak = simind.get_energy_window_width(headerfiles_photopeak[0])
            ww_lower = simind.get_energy_window_width(headerfiles_lower[0])
            ww_upper = simind.get_energy_window_width(headerfiles_upper[0])
            lower_scatter = projections_realization[1].unsqueeze(0)
            upper_scatter = projections_realization[2].unsqueeze(0)
            scatter = (lower_scatter/ww_lower+upper_scatter/ww_upper)*ww_peak/2

            # scatter noiseless (ground truth)
            lower_scatter_noiseless = projections[1].unsqueeze(0) * dT
            upper_scatter_noiseless = projections[2].unsqueeze(0) * dT
            scatter_noiseless = (lower_scatter_noiseless/ww_lower+upper_scatter_noiseless/ww_upper)*ww_peak/2

            print(f'Run: {run}')
            num_channels = [int(i) for i in exp_code.split('_')[1].split('x')]
            print(f'Initializing neural network config: {num_channels}')
            unet = UNetCustom(n_channels=num_channels)
            # torch.manual_seed(42)
            truth = scatter[0].unsqueeze(1)
            input = torch.randn(truth.shape)
            torch.save(input, 'random_noise_input.pt')
            optimizer_lfbgs = LBFGS(unet.parameters(), lr=1, max_iter=20, history_size=100)
            criterion = torch.nn.MSELoss()
            unet.train()
            n_epochs = 10

            # initialize folders
            print(f'Creating folders')
            logs_dir = os.path.join(save_dir, 'logs', f'{exp_code}')
            models_dir = os.path.join(save_dir, 'models', f'{exp_code}')
            preds_dir = os.path.join(save_dir, 'predictions', f'{exp_code}')
            recons_dir = os.path.join(save_dir, 'reconstructions', f'{exp_code}')
            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(preds_dir, exist_ok=True)
            os.makedirs(recons_dir, exist_ok=True)

            # training for n_epochs epochs
            def get_closure(optimizer):
                optimizer.zero_grad()
                NM_prediction = unet(input)
                loss = criterion(NM_prediction, truth)
                loss.backward()
                return loss

            print(f'Run {run}: Training the network')
            loss_list = []
            for epoch in range(n_epochs):  
                loss = optimizer_lfbgs.step(lambda: get_closure(optimizer_lfbgs))
                loss_list.append(loss.item())
                model_save_fpath = os.path.join(models_dir, f'run{pad_zeros_at_front(run, 2)}_checkpoint={pad_zeros_at_front(epoch+1, 2)}.pth')
                torch.save(unet.state_dict(), model_save_fpath)
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            logs_fpath = os.path.join(logs_dir, f'run{pad_zeros_at_front(run, 2)}_logs.csv')
            print(f'Run {run}: Saving training logs at: {logs_fpath}')
            df = pd.DataFrame(loss_list, columns=['Loss'])
            df.to_csv(logs_fpath, index=False)

            # prediction
            print(f'Run {run}: Saving predicted scatter at: {logs_fpath}')
            scatter_pred = unet(input)
            scatter_pred = scatter_pred.swapaxes(1,0)
            scatter_fpath = os.path.join(preds_dir, f'run{pad_zeros_at_front(run, 2)}_pred.pt')
            torch.save(scatter_pred, scatter_fpath)

            # image reconstruction
            print(f'Run {run}: Initializing reconstruction objects')
            object_meta, proj_meta = simind.get_metadata(headerfiles_photopeak[0])
            attenuation_path = os.path.join(path, 'multi_projections', 'mu208.hct')
            attenuation_map = simind.get_attenuation_map(attenuation_path)
            att_transform = SPECTAttenuationTransform(attenuation_map)
            psf_meta = simind.get_psfmeta_from_header(headerfiles_photopeak[0])
            psf_transform = SPECTPSFTransform(psf_meta)

            system_matrix = SPECTSystemMatrix(
                obj2obj_transforms = [att_transform,psf_transform],
                proj2proj_transforms = [],
                object_meta = object_meta,
                proj_meta = proj_meta,
                n_parallel=4)

            photopeak = projections_realization[0].unsqueeze(0)
            likelihood_scatter_noiseless = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter_noiseless)
            likelihood_scatter_noisy = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)
            likelihood_scatter_pred = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter_pred)

            print(f'Run {run}: Reconstructing with noiseless scatter')
            recon_algo_scatter_noiseless = OSEM(likelihood_scatter_noiseless)
            recon_osem_scatter_noiseless = recon_algo_scatter_noiseless(n_iters = 4, n_subsets = 8)
            savepath_scatter_noiseless = os.path.join(recons_dir, f'run{pad_zeros_at_front(run, 2)}_recon_osem_scatter_noiseless.pt')
            torch.save(recon_osem_scatter_noiseless, savepath_scatter_noiseless)

            print(f'Run {run}: Reconstructing with noisy scatter')
            recon_algo_scatter_noisy = OSEM(likelihood_scatter_noisy)
            recon_osem_scatter_noisy = recon_algo_scatter_noisy(n_iters = 4, n_subsets = 8)
            savepath_scatter_noisy = os.path.join(recons_dir, f'run{pad_zeros_at_front(run,2)}_recon_osem_scatter_noisy.pt')
            torch.save(recon_osem_scatter_noisy, savepath_scatter_noisy)

            print(f'Run {run}: Reconstructing with predicted scatter')
            recon_algo_scatter_pred = OSEM(likelihood_scatter_pred)
            recon_osem_scatter_pred = recon_algo_scatter_pred(n_iters = 4, n_subsets = 8)
            savepath_scatter_pred = os.path.join(recons_dir, f'run{pad_zeros_at_front(run,2)}_recon_osem_scatter_pred.pt')
            torch.save(recon_osem_scatter_pred, savepath_scatter_pred)

            print(f'Run {run}: Clearing GPU memory')
            del unet, truth, input, optimizer_lfbgs, criterion, loss_list, scatter_pred
            torch.cuda.empty_cache()
# %%
elapsed = time.time() - start_time 
print(f'Total time taken: {elapsed/60:.2f} min')