from functions import *

hps = {'sep_lr': 1.0, 'zero_lr': 0.09, 'hidden': 208, 'channel_index': 4, 'norm_type': 'group_norm', 'weight_decay': 1e-05, 'sep_norm': 'L2', 'batch_size': 16, 'lr': 0.1}
channel_options = [[8, 16], [8, 16, 32], [8, 16, 32, 64], [8, 16, 32, 64, 128], [24, 48, 96, 144]]

model, _, _ = train(CircleTriangleDataset(), batch_size=64, visualise=True, max_epochs=200, test_save_step=20)
#model, _, _ = train(CircleTriangleDataset(), batch_size=hps['batch_size'], channels=channel_options[hps['channel_index']], sep_lr=hps['sep_lr'], zero_lr=hps['zero_lr'], hidden=hps['hidden'], norm_type=hps['norm_type'], weight_decay=hps['weight_decay'], sep_norm=hps['sep_norm'], lr=hps['lr'], visualise=True, max_epochs=100, test_save_step=5)
