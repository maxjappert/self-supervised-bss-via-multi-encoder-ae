import pickle
import sys

from functions import TwoSourcesDataset, train

debug = False

dataset_train = TwoSourcesDataset(debug=debug, split='train', name='toy_dataset', reduction_ratio=0.001)
dataset_val = TwoSourcesDataset(debug=debug, split='val', name='toy_dataset', reduction_ratio=0.001)

if len(sys.argv) > 1:
    original = sys.argv[1] == 'original'
    linear = sys.argv[1] == 'linear'

    name = f'toy_separator' + ('_original' if original else '_linear' if linear else '')
else:
    linear = False
    original = False
    name = f'debug'

print(f'Training {name}')

model, train_losses, val_losses = train(dataset_train,
                                        dataset_val,
                                        num_encoders=2,
                                        image_height=64,
                                        image_width=64,
                                        visualise=True,
                                        kernel_size=7,
                                        linear=linear,
                                        name=name,
                                        original_implementation=original,
                                        compute_sdr=False
                                        )

output = {'train_losses': train_losses,
          'val_losses': val_losses}

with open(f'results/{name}.pkl', 'wb') as f:
    pickle.dump(output, f)
