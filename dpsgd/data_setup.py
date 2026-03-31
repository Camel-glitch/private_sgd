from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms



def get_dataloaders(DATASET='mnist', BATCH_SIZE=64):
    transform = transforms.Compose([transforms.ToTensor()])
    
    if DATASET == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # Split 70% Train / 30% Valid
    train_size = int(0.7 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, valid_loader, test_loader