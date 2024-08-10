import os
import random
import sys

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import NMF
import soundfile as sf

from evaluation_metric_functions import compute_spectral_metrics
from functions import evaluate_separation_ability, create_combined_image, metric_index_mapping

def nmf_approx_two_sources(S_mix_gt):
    if len(S_mix_gt.shape) == 3:
        S_mix_gt = np.mean(S_mix_gt, axis=2)

    nmf = NMF(n_components=2, random_state=0)
    W = nmf.fit_transform(S_mix_gt.cpu())
    H = nmf.components_

    # Reconstruct sources
    S1_approx = np.dot(W[:, 0:1], H[0:1, :])
    S2_approx = np.dot(W[:, 1:2], H[1:2, :])

    return S1_approx, S2_approx
