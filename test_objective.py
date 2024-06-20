import random

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.draw import disk
import matplotlib.pyplot as plt

from functions import evaluate_separation_ability


# Create synthetic images for testing
def create_image(size, center, radius):
    image = np.zeros(size, dtype=np.uint8)
    rr, cc = disk(center, radius, shape=size)
    image[rr, cc] = 255
    return image

print(np.array(Image.open('pngs/third_spectro_musdb18_linear_0.png').convert('L')).shape)

# Create approximations and ground truths
size = (100, 100)
approxs = [
    np.array(Image.open('pngs/third_spectro_musdb18_linear_0.png').convert('L')),
    np.array(Image.open('pngs/third_spectro_musdb18_linear_1.png').convert('L')),
    np.array(Image.open('pngs/third_spectro_musdb18_linear_mix.png').convert('L'))
]

gts = [
    np.array(Image.open('pngs/third_spectro_musdb18_linear_0_gt.png').convert('L')),
    np.array(Image.open('pngs/third_spectro_musdb18_linear_1_gt.png').convert('L')),
    np.array(Image.open('pngs/third_spectro_musdb18_linear_mix_gt.png').convert('L'))
]

random.shuffle(approxs)
random.shuffle(gts)

# Plot images for visual verification
fig, axes = plt.subplots(2, 3, figsize=(10, 5))
for i, (approx, gt) in enumerate(zip(approxs, gts)):
    axes[0, i].imshow(approx, cmap='gray')
    axes[0, i].set_title(f'Approx {i+1}')
    axes[0, i].axis('off')

    axes[1, i].imshow(gt, cmap='gray')
    axes[1, i].set_title(f'GT {i+1}')
    axes[1, i].axis('off')

plt.show()

# Evaluate the separation ability
separation_ability = evaluate_separation_ability(approxs, gts)
print(f'Separation Ability: {separation_ability}')
