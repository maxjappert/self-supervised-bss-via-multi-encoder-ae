import argparse
import pickle
import sys

from functions import TwoSourcesDataset, train

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--name')      # option that takes a value
parser.add_argument('--linear',
                    action='store_true')  # on/off flag

args = parser.parse_args()

# todo: remove
# args.linear = False
# args.name = 'debug'

debug = False

dataset_name = 'toy_dataset' if args.name.__contains__('toy') else 'musdb_18_prior'

dataset_train = TwoSourcesDataset(debug=debug, split='train', name=dataset_name, reduction_ratio=0.001)
dataset_val = TwoSourcesDataset(debug=debug, split='val', name=dataset_name, reduction_ratio=0.001)

name = args.name + ('_linear' if args.linear else '')


print(f'Training {name}')

model, train_losses, val_losses = train(dataset_train,
                                        dataset_val,
                                        num_encoders=2,
                                        image_height=64,
                                        image_width=64,
                                        visualise=True,
                                        kernel_size=7,
                                        linear=args.linear,
                                        name=name,
                                        original_implementation=False,
                                        compute_sdr=False,
                                        hidden=196
                                        )

output = {'train_losses': train_losses,
          'val_losses': val_losses}

with open(f'results/{name}.pkl', 'wb') as f:
    pickle.dump(output, f)
