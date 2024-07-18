import numpy as np
import torch

image_h = 1024
image_w = 128


def g(x):
    return np.sum(x, axis=0)


L = 10
T = 1000
k = 4
batch_size = 4
x = torch.rand(k, image_h, image_w, requires_grad=True)
alpha = torch.ones(k) * (1.0/k)
delta = 0.1
m = ...
vae = ...

sigma_start = 0.1
sigma_end = 1
sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                        end=torch.log10(torch.tensor(sigma_end)),
                        steps=L)

x_chain = []

for i in range(L):
    eta_i = delta * sigmas[i]**2 / sigmas[L]**2

    for t in range(T):
        epsilon_t = torch.randn(x.shape)
        elbo = vae.log_prob(x).mean(axis=0).squeeze()
        grad_log_p_x = torch.autograd.grad(elbo, x, retain_graph=True)[0]
        u = x + eta_i * vae.log_prob(x) + grad_log_p_x + torch.sqrt(2*eta_i) * epsilon_t
        x = u - (eta_i / sigmas[t]**2) * torch.diag(alpha) * (m - g(x))
        x_chain.append(x)

