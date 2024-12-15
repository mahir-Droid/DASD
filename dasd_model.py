import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List

class StochasticFeatureMask(nn.Module):
    def __init__(self, feature_dim: int, mask_ratio: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            mask = (torch.rand_like(x) > self.mask_ratio).float()
            return x * mask
        else:
            outputs = []
            for _ in range(5):  # Ensemble of 5 different masks
                mask = (torch.rand_like(x) > self.mask_ratio).float()
                outputs.append(x * mask)
            return torch.mean(torch.stack(outputs), dim=0)

class DensityRegulator(nn.Module):
    def __init__(self, feature_dim: int, noise_scale: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_scale = noise_scale
        
    def forward(self, x: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        # Add more noise to high-density regions
        noise_level = self.noise_scale * density.view(-1, 1, 1, 1)
        noise = torch.randn_like(x) * noise_level
        return x + noise

class DensityAwareBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.feature_mask = StochasticFeatureMask(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.feature_mask(out, self.training)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class DASDNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.density_regulator = DensityRegulator(3)
        
        # Feature extraction blocks
        self.block1 = DensityAwareBlock(3, 64)
        self.block2 = DensityAwareBlock(64, 128)
        self.block3 = DensityAwareBlock(128, 256)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Density estimation parameters
        self.register_buffer('class_means', torch.zeros(num_classes, 256))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
    def compute_density(self, features: torch.Tensor) -> torch.Tensor:
        # Compute local density using gaussian kernel
        dists = torch.cdist(features, features)
        sigma = torch.median(dists)
        density = torch.mean(torch.exp(-dists / (2 * sigma**2)), dim=1)
        return density
    
    def update_class_stats(self, features: torch.Tensor, labels: torch.Tensor):
        for c in range(self.class_means.shape[0]):
            mask = (labels == c)
            if torch.any(mask):
                class_features = features[mask]
                self.class_means[c] = (self.class_means[c] * self.class_counts[c] + 
                                     torch.sum(class_features, dim=0)) / (self.class_counts[c] + len(class_features))
                self.class_counts[c] += len(class_features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply density regulation during training
        if self.training and labels is not None:
            features = self.get_features(x)
            density = self.compute_density(features)
            x = self.density_regulator(x, density)
            
        # Feature extraction
        features = self.get_features(x)
        
        # Update class statistics during training
        if self.training and labels is not None:
            self.update_class_stats(features, labels)
            
        # Classification
        logits = self.fc(features)
        
        # Compute density for detection
        density = self.compute_density(features)
        
        return logits, density

def detect_adversarial(model: nn.Module, inputs: torch.Tensor, threshold: float = 0.85) -> torch.Tensor:
    """Detect potential adversarial examples based on density."""
    model.eval()
    with torch.no_grad():
        _, density = model(inputs)
    return density > threshold

# Training utilities
class DensityAwareLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, density_weight: float = 0.1):
        super().__init__()
        self.base_criterion = base_criterion
        self.density_weight = density_weight
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        # Classification loss
        base_loss = self.base_criterion(logits, labels)
        
        # Density regularization - penalize high densities
        density_loss = torch.mean(torch.square(density))
        
        return base_loss + self.density_weight * density_loss