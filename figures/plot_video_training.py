import pickle
from matplotlib import pyplot as plt

# this isn't fabricated data, i just appended the accuracies list rather than the actual accuracy in the train function

split = 'train'

if split == 'val':
    raft_resnet_accs = [
        49.3174, 61.0922, 57.8498, 61.4334, 66.5529, 67.5768, 67.2355, 67.0648,
        69.4539, 67.5768, 68.9420, 68.0887, 71.5017, 69.6246, 72.3549, 71.3311,
        70.1365, 74.0614, 73.2082, 74.7440, 71.6724, 72.6962, 72.1843, 71.6724,
        76.2799, 73.2082, 75.7679, 73.0375, 71.8430, 72.5256, 73.7201, 74.7440,
        73.8908, 75.2560, 76.1092, 72.0137, 77.8157, 75.4266, 71.3311, 73.5495,
        76.2799, 75.5973, 73.7201, 77.1331, 73.7201, 77.1331, 72.8669, 74.9147,
        72.3549, 75.7679
    ]

    raft_accs = [
        51.0239, 53.2423, 57.1672, 57.3379, 59.0444, 59.8976, 63.9932, 63.1399,
        58.8737, 61.9454, 64.3345, 64.5051, 66.3823, 63.9932, 65.8703, 62.9693,
        63.6519, 68.2594, 66.8942, 65.5290, 69.1126, 68.0887, 68.6007, 66.0410,
        67.7474, 65.5290, 66.3823, 64.5051, 65.0171, 65.3584, 69.1126, 67.0648,
        66.2116, 66.7235, 63.1399, 66.3823, 68.7713, 68.6007, 65.0171, 67.4061,
        66.5529, 66.8942, 66.5529, 67.2355, 68.2594, 69.1126, 63.6519, 63.1399,
        65.5290, 69.4539
    ]

    resnet_accs = [
        52.0478, 57.6792, 58.3618, 58.1911, 66.3823, 63.1399, 68.2594, 66.7235,
        73.8908, 67.0648, 67.7474, 72.6962, 69.9659, 72.0137, 69.2833, 73.7201,
        70.4778, 70.4778, 66.5529, 66.3823, 70.4778, 68.2594, 70.3072, 68.4300,
        76.9625, 71.8430, 73.0375, 69.9659, 69.9659, 65.8703, 68.6007, 72.5256,
        69.9659, 69.6246, 71.1604, 72.8669, 71.8430, 72.1843, 72.8669, 70.6485,
        71.5017, 74.4027, 69.9659, 69.6246, 75.5973, 71.3311, 67.9181, 71.5017,
        71.3311
    ]



    simple_accs = [
        50.3413, 53.4130, 49.6587, 53.7543, 52.3891, 48.4642, 53.4130, 51.3652,
        57.8498, 56.9966, 59.0444, 57.8498, 55.9727, 60.7509, 60.0683, 63.6519,
        63.4812, 62.9693, 66.8942, 66.8942, 63.6519, 66.0410, 63.8225, 63.8225,
        65.1877
    ]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(raft_resnet_accs) + 1), raft_resnet_accs, label='RAFT + ResNet', marker='o')
    plt.plot(range(1, len(resnet_accs) + 1), resnet_accs, label='No RAFT + ResNet', marker='s')
    plt.plot(range(1, len(raft_accs) + 1), raft_accs, label='RAFT + Simple CNN', marker='^')
    plt.plot(range(1, len(simple_accs) + 1), simple_accs, label='No RAFT + Simple CNN', marker='d')

    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Accuracy in %')
    plt.title('Validation Accuracies Over Epochs for Multi-Modal Classifier Setups')
    plt.legend()
    plt.grid(True)
    plt.savefig('video_train_plots_val.png', dpi=300, bbox_inches='tight')
elif split == 'train':
    raft_resnet_accs = pickle.load(open('../results/video_model_raft_resnet_train_accuracies.pkl', 'rb'))
    raft_accs = pickle.load(open('../results/video_model_raft_train_accuracies.pkl', 'rb'))
    resnet_accs = pickle.load(open('../results/video_model_resnet_train_accuracies.pkl', 'rb'))
    simple_accs = pickle.load(open('../results/video_model_simple_train_accuracies.pkl', 'rb'))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(raft_resnet_accs) + 1), raft_resnet_accs, label='RAFT + ResNet', marker='o')
    plt.plot(range(1, len(resnet_accs) + 1), resnet_accs, label='No RAFT + ResNet', marker='s')
    plt.plot(range(1, len(raft_accs) + 1), raft_accs, label='RAFT + Simple CNN', marker='^')
    plt.plot(range(1, len(simple_accs) + 1), simple_accs, label='No RAFT + Simple CNN', marker='d')

    plt.xlabel('Training Epochs')
    plt.ylabel('Train Accuracy in %')
    plt.title('Train Accuracies Over Epochs for Multi-Modal Classifier Setups')
    plt.legend()
    plt.grid(True)
    plt.savefig('video_train_plots_train.png', dpi=300, bbox_inches='tight')
