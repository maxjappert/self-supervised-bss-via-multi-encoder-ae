from functions import *

print('lesgoo')

model = get_model()
dataloader = get_dataloader()

train(model, dataloader, max_epochs=50)
