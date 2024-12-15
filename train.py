import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from dasd_model import DASDNet, DensityAwareLoss
import matplotlib.pyplot as plt

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model initialization
    model = DASDNet().to(device)
    
    # Loss and optimizer
    criterion = DensityAwareLoss(nn.CrossEntropyLoss())
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Training loop
    epochs = 200
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, density = model(inputs, labels)
            loss = criterion(logits, labels, density)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        adv_detected = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, density = model(inputs)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                adv_detected += torch.sum(density > 0.85).item()
        
        accuracy = 100. * correct / total
        test_accuracies.append(accuracy)
        train_losses.append(running_loss / len(trainloader))
        
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.3f} '
              f'Accuracy: {accuracy:.2f}% Potential adversarials: {adv_detected}')
        
        scheduler.step()
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    
    # Save model
    torch.save(model.state_dict(), 'dasd_model.pth')
    
if __name__ == '__main__':
    train_model()