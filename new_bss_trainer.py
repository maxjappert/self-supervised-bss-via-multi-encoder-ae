import sys

from functions import TwoSourcesDataset, train

debug = False

dataset_train = TwoSourcesDataset(debug=debug, split='train', name='toy_dataset')
dataset_val = TwoSourcesDataset(debug=debug, split='val', name='toy_dataset')

model, train_losses, val_losses = train(dataset_train,
                                        dataset_val,
                                        num_encoders=2,
                                        image_height=64,
                                        image_width=64,
                                        visualise=True,
                                        kernel_size=7,
                                        linear=False,
                                        name='toy_separator',
                                        original_implementation=sys.argv[1] == 'original'
                                        )
