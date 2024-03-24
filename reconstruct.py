#%%
from __future__ import annotations
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
# %%
path = '/home/sahamed/Projects/pytomography/scatter-estimation-deep-image-prior/simind_tutorial'

organs = ['bkg', 'liver', 'l_lung', 'r_lung', 'l_kidney', 'r_kidney','salivary', 'bladder']
activities = [2500, 450, 7, 7, 100, 100, 20, 90] # MBq
headerfiles = [os.path.join(path, 'multi_projections', organ, 'photopeak.h00') for organ in organs]
headerfiles_lower = [os.path.join(path, 'multi_projections', organ, 'lowerscatter.h00') for organ in organs]
headerfiles_upper = [os.path.join(path, 'multi_projections', organ, 'upperscatter.h00') for organ in organs]

projections = simind.get_projections([headerfiles, headerfiles_lower, headerfiles_upper], weights=activities)
