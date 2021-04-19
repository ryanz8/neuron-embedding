import torch
import torchvision
import torchvision.transforms as transforms

def CIFAR():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(data, [40000, 10000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000,
                                            shuffle=False, num_workers=6, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader


def MNIST():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

    data = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(data, [50000, 10000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())

    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000,
                                            shuffle=False, num_workers=6, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader
