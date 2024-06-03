from functions import *

model, _, _ = train(get_model(channels=[24, 48, 96, 144], hidden=96, norm_type='group_norm'), CircleTriangleDataset(), batch_size=1024, sep_norm='L1', lr=0.001, weight_decay=1e-05, sep_lr=0.5, zero_lr=0.01, z_decay=0.01, max_epochs=100, name='previously_best')

print(test(model, CircleTriangleDataset(), visualise=True))
