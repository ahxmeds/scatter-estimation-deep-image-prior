#%%
# from __future__ import annotations
import os
from pytomography.io.SPECT import simind
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM, OSMAPOSL, BSREM, KEM
from pytomography.transforms.shared import GaussianFilter
from torch import poisson
import matplotlib.pyplot as plt
import pytomography
import torch
from pytomography.callbacks import Callback
from pytomography.priors import RelativeDifferencePrior
from pytomography.priors import TopNAnatomyNeighbourWeight
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import KEMTransform
from pytomography.projectors.shared import KEMSystemMatrix
import numpy as np 
# %%
path = '/home/sahamed/Projects/pytomography/dip_scatter/simind_tutorial'
organs = ['bkg', 'liver', 'l_lung', 'r_lung', 'l_kidney', 'r_kidney','salivary', 'bladder']
activities = [2500, 450, 7, 7, 100, 100, 20, 90] # MBq
headerfiles_photopeak = [os.path.join(path, 'multi_projections', organ, 'photopeak.h00') for organ in organs]
headerfiles_lower = [os.path.join(path, 'multi_projections', organ, 'lowerscatter.h00') for organ in organs]
headerfiles_upper = [os.path.join(path, 'multi_projections', organ, 'upperscatter.h00') for organ in organs]

projections = simind.get_projections([headerfiles_photopeak, headerfiles_lower, headerfiles_upper], weights=activities)

# %%
ww_photopeak = simind.get_energy_window_width(headerfiles_photopeak[0])
ww_lower = simind.get_energy_window_width(headerfiles_lower[0])
ww_upper = simind.get_energy_window_width(headerfiles_upper[0])
dT = 15 # seconds per projection
projections_realization = torch.poisson(projections * dT)
photopeak = projections_realization[0].unsqueeze(0)
lower_scatter = projections_realization[1].unsqueeze(0)
upper_scatter = projections_realization[2].unsqueeze(0)
scatter = (lower_scatter/ww_lower + upper_scatter/ww_upper)*(ww_photopeak/2)

# %%
object_meta, proj_meta = simind.get_metadata(headerfiles_photopeak[0])

# %%
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
likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)

# %%
###################################################################
##################                               ##################
##################             OSEM              ##################
##################                               ##################
###################################################################

recon_algorithm = OSEM(likelihood)
recon_OSEM = recon_algorithm(n_iters = 4, n_subsets = 8)

# %%
###################################################################
##################                               ##################
##################           OSMAPOSL            ##################
##################                               ##################
###################################################################
weight_top8anatomy = TopNAnatomyNeighbourWeight(attenuation_map, N_neighbours=8)
prior_rdpap = RelativeDifferencePrior(beta=0.3, gamma=2, weight=weight_top8anatomy)
recon_algorithm_osmaposl = OSMAPOSL(
    likelihood = likelihood,
    prior = prior_rdpap)
recon_osmaposl = recon_algorithm_osmaposl(n_iters = 40, n_subsets = 8)
# %%
###################################################################
##################                               ##################
##################              BSREM            ##################
##################                               ##################
###################################################################
weight_top8anatomy = TopNAnatomyNeighbourWeight(attenuation_map, N_neighbours=8)
prior_rdpap = RelativeDifferencePrior(beta=0.3, gamma=2, weight=weight_top8anatomy)
recon_algorithm_bsrem = BSREM(
    likelihood = likelihood,
    prior = prior_rdpap,
    relaxation_sequence = lambda n: 1/(n/50+1))
recon_bsrem = recon_algorithm_bsrem(40,8)
# %%
fig, ax = plt.subplots(2, 3, figsize=(5,9))
cmap_val = 'nipy_spectral'
ax[0][0].imshow(np.rot90(recon_OSEM[0].cpu()[:,70]), cmap=cmap_val)
ax[0][1].imshow(np.rot90(recon_osmaposl[0].cpu()[:,70]), cmap=cmap_val)
ax[0][2].imshow(np.rot90(recon_bsrem[0].cpu()[:,70]), cmap=cmap_val)
ax[1][0].imshow(np.rot90(recon_OSEM[0].cpu()[70]), cmap=cmap_val)
ax[1][1].imshow(np.rot90(recon_osmaposl[0].cpu()[70]), cmap=cmap_val)
ax[1][2].imshow(np.rot90(recon_bsrem[0].cpu()[70]), cmap=cmap_val)
ax[0][0].set_title('OSEM')
ax[0][1].set_title('OSMAPOSL')
ax[0][2].set_title('BSREM')
for i in range(2):
    for j in range(3):
        ax[i][j].axis('off')
plt.show()
plt.close('all')

# plt.subplots(1,2,figsize=(8,6))
# plt.subplot(121)
# plt.pcolormesh(recon_bsrem[0].cpu()[:,70].T, cmap='magma')
# plt.colorbar()
# plt.axis('off')
# plt.title('Coronal Slice')
# plt.subplot(122)
# plt.pcolormesh(recon_bsrem[0].cpu()[70].T, cmap='magma')
# plt.colorbar()
# plt.axis('off')
# plt.title('Sagittal Slice')
# plt.show()
# %%
