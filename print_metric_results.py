import numpy as np

vae_name = 'musdb'

results_basis = np.load(f'results/results_basis_{vae_name}.npy')
results_basis_finetuned = np.load(f'results/results_basis_finetuned_{vae_name}.npy')
results_bss = np.load(f'results/results_bss_{vae_name}.npy')
results_bss_linear = np.load(f'results/results_bss_linear_{vae_name}.npy')
results_noise = np.load(f'results/results_noise_{vae_name}.npy')
results_prior_samples = np.load(f'results/results_prior_samples_{vae_name}.npy')
results_nmf = np.load(f'results/results_nmf_{vae_name}.npy')

results = {'basis': results_basis,
           'basis_finetuned': results_basis_finetuned,
           'bss': results_bss,
           'bss_linear': results_bss_linear,
           'noise': results_noise,
           'prior_samples': results_prior_samples,
           'nmf': results_nmf}

print(np.mean(results_basis, axis=0).shape)

metrics = ['sdr', 'isr', 'sir', 'sar']

for i, metric in enumerate(metrics):
    for result_type in results.keys():
        result = np.mean(results[result_type], axis=0)
        std_e = np.std(result[i], ddof=1) / np.sqrt(len(result[i]))

        print(f'{result_type} {metric}: {np.round(np.mean(result[i], axis=0), 1)} +- {np.round(std_e, 2)}')


