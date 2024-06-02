from functions import *

print('lesgoo')

model = get_model()

model = train(model, CircleTriangleDataset(), batch_size=1024, max_epochs=10, name='model')
print('here')
test(model, CircleTriangleDataset())
