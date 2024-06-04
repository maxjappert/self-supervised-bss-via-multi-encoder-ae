from functions import *

#print(len(CircleTriangleDataset()))

#hp = {'lr': 0.00016062499813462855, 'weight_decay': 0.00039653179743469276, 'z_decay': 1.0277939180373177e-10, 'sep_lr': 0.0013962320530318257, 'zero_lr': 9.929284549910692e-05, 'hidden': 1696, 'channel_index': 5, 'sep_norm': 'L1', 'norm_type': 'none'}

#channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128]]

model, _, _ = train(CircleTriangleDataset(), channels=[16, 32, 64, 128, 256], hidden=96, norm_type='none', batch_size=256, sep_norm='L2', lr=0.001, weight_decay=1e-05, sep_lr=0.8, zero_lr=0.1, z_decay=0.01, max_epochs=100, name='new_attempt', visualise=False, linear=False)

print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_alt_1'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_alt_2'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_alt_3'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_alt_4'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_alt_5'))

model, _, _ = train(CircleTriangleDataset(), channels=[24, 48, 96, 144], hidden=96, norm_type='group_norm', batch_size=1024, sep_norm='L1', lr=0.001, weight_decay=1e-05, sep_lr=0.5, zero_lr=0.01, z_decay=0.01, max_epochs=100, name='previously_best', visualise=False, linear=False)
#model, _, _ = train(CircleTriangleDataset(), channels=[24, 48, 96, 144], hidden=96, norm_type='none', batch_size=1024, sep_norm='L1', lr=0.001, weight_decay=0, sep_lr=1, zero_lr=0.1, z_decay=0, max_epochs=100, visualise=False, linear=False)
#model, _, _ = train(CircleTriangleDataset(), channels=[24, 48, 96, 144], hidden=128, norm_type='none', batch_size=1024, sep_norm='L2', lr=0.0001, weight_decay=1e-05, sep_lr=0.5, zero_lr=0.01, z_decay=0.01, max_epochs=50, visualise=False, linear=True)
#model, _, _ = train(CircleTriangleDataset(), channels=channel_options[hp['channel_index']], hidden=hp['hidden'], norm_type=hp['norm_type'], batch_size=512, sep_norm=hp['sep_norm'], lr=hp['lr'], weight_decay=hp['weight_decay'], sep_lr=hp['sep_lr'], zero_lr=hp['zero_lr'], z_decay=hp['z_decay'], max_epochs=100, visualise=False)

#model = get_model(channels=[24, 48, 96, 144], hidden=96, norm_type='none')
#model.load_state_dict(torch.load('outputs/2024-03-12/14-40-16/logs/my_experiment/version_0/checkpoints/last.ckpt'))

print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test1'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test2'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test3'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test4'))
print(test(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test5'))

model, _, _ = train(CircleTriangleDataset(), channels=[16, 32, 64, 128], hidden=128, norm_type='none', batch_size=1024, sep_norm='L2', lr=0.0001, weight_decay=1e-05, sep_lr=0.5, zero_lr=0.01, z_decay=0.01, max_epochs=100, visualise=False, linear=True)

visualise_linear(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_linear1')
visualise_linear(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_linear2')
visualise_linear(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_linear3')
visualise_linear(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_linear4')
visualise_linear(model, CircleTriangleDataset(), visualise=True, num_samples=1, name='test_linear5')
