from functions import *

#print(len(CircleTriangleDataset()))

channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128]]

model, _, _ = train(CircleTriangleDataset(), channels=[24, 48, 96, 144], hidden=96, norm_type='group_norm', batch_size=1024, sep_norm='L1', lr=0.001, weight_decay=1e-05, sep_lr=0.5, zero_lr=0.01, z_decay=0.01, max_epochs=100, name='previously_best', visualise=True)
#model, _, _ = train(CircleTriangleDataset(), channels=[24, 48, 96, 144], hidden=4064, norm_type='group_norm', batch_size=512, sep_norm='L2', lr=3.4789328616807426e-05, weight_decay=0.0001258303842780681, sep_lr=1.6288000973950926e-06, zero_lr=6.349763823589664e-08, z_decay=1.6041337546222484e-10, max_epochs=100, visualise=True)

print(test(model, CircleTriangleDataset(), visualise=True))
