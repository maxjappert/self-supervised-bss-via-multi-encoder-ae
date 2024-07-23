from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from functions_prior import train_vae

class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

image_h = 64
image_w = 64

transform = transforms.Compose([
    transforms.Resize((image_h, image_w)),
    transforms.ToTensor(),
    MinMaxNormalize()
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

train_vae(train_loader,
          test_loader, latent_dim=4, image_h=image_h, image_w=image_w, channels=[4, 8, 16], visualise=True, name='mnist_large')
