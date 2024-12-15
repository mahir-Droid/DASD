import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dasd_model import DASDNet, detect_adversarial
import matplotlib.pyplot as plt
import numpy as np

def evaluate_defense():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model = DASDNet().to(device)
    model.load_state_dict(torch.load('dasd_model.pth'))
    model.eval()
    
    # Prepare test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Evaluate clean accuracy and detection rates
    correct = 0
    total = 0
    densities = []
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, density = model(inputs)
            _, predicted = logits.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            densities.extend(density.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    densities = np.array(densities)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Plot density distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(densities, bins=50, alpha=0.7)
    plt.title('Density Distribution')
    plt.xlabel('Density')
    plt.ylabel('Count')
    
    # Plot density vs accuracy
    density_thresholds = np.linspace(0, np.max(densities), 100)
    accuracies = []
    detection_rates = []
    
    for threshold in density_thresholds:
        mask = densities < threshold
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == true_labels[mask]) * 100
            accuracies.append(acc)
            detection_rates.append(np.mean(mask) * 100)
    
    plt.subplot(1, 2, 2)
    plt.plot(detection_rates, accuracies)
    plt.title('Accuracy vs Detection Rate')
    plt.xlabel('Detection Rate (%)')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('defense_evaluation.png')
    
    print(f'Clean Accuracy: {accuracy:.2f}%')
    print(f'Average Density: {np.mean(densities):.3f}')
    print(f'Density Std: {np.std(densities):.3f}')

if __name__ == '__main__':
    evaluate_defense()