from functions import *

model, _, _ = train(CircleTriangleDataset(), batch_size=64, visualise=True, max_epochs=25, test_save_step=5)
