import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'darknet-pytorch'))

import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import numpy as np
from model import darknet53

class HybridBrainTumorModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridBrainTumorModel, self).__init__()
        
        # Initialize the three base models
        # 1. DarkNet53 (using official implementation)
        self.darknet = darknet53(num_classes=1000, init_weight=True)
        
        # 2. EfficientNet-B0
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 3. DenseNet201
        self.densenet = models.densenet201(pretrained=True)
        
        # Modify activation functions to avoid in-place operations
        self._modify_activations(self.darknet)
        self._modify_activations(self.efficientnet)
        self._modify_activations(self.densenet)
        
        # Modify the final layers to extract features
        self._modify_base_models()
        
        # Feature dimensions from each model (actual output sizes)
        self.darknet_features = 1024    # DarkNet53's last conv layer channels
        self.efficientnet_features = 1280  # EfficientNet-B0's feature size
        self.densenet_features = 1920   # DenseNet201's feature size
        
        # Combined feature dimension
        total_features = self.darknet_features + self.efficientnet_features + self.densenet_features
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 500),  # Reduced features as per paper
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(500, num_classes)
        )

    def _modify_activations(self, model):
        """Modify all activation functions to not use in-place operations"""
        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                module.inplace = False

    def _modify_base_models(self):
        # Store the original classifiers
        self.darknet_classifier = self.darknet.classifier
        self.efficientnet_classifier = self.efficientnet._fc
        self.densenet_classifier = self.densenet.classifier
        
        # Remove the original classifiers
        self.darknet.classifier = nn.Identity()
        self.efficientnet._fc = nn.Identity()
        self.densenet.classifier = nn.Identity()

    def forward(self, x):
        # Resize input for each model
        x_darknet = nn.functional.interpolate(x, size=(256, 256))
        x_efficient = nn.functional.interpolate(x, size=(224, 224))
        x_dense = nn.functional.interpolate(x, size=(224, 224))
        
        # Get features from DarkNet53
        darknet_features = self.darknet.features(x_darknet)
        darknet_features = nn.functional.adaptive_avg_pool2d(darknet_features, (1, 1))
        darknet_features = darknet_features.view(darknet_features.size(0), -1)
        
        # Get features from EfficientNet
        efficient_features = self.efficientnet(x_efficient)
        
        # Get features from DenseNet
        dense_features = self.densenet(x_dense)
        
        # Concatenate features
        combined_features = torch.cat([
            darknet_features,
            efficient_features,
            dense_features
        ], dim=1)
        
        # Final classification
        return self.classifier(combined_features)

    def get_features(self, x):
        """Extract features from all three models separately"""
        # Resize input for each model
        x_darknet = nn.functional.interpolate(x, size=(256, 256))
        x_efficient = nn.functional.interpolate(x, size=(224, 224))
        x_dense = nn.functional.interpolate(x, size=(224, 224))
        
        # Get features from each model
        darknet_features = self.darknet(x_darknet)
        efficient_features = self.efficientnet(x_efficient)
        dense_features = self.densenet(x_dense)
        
        return {
            'darknet': darknet_features,
            'efficientnet': efficient_features,
            'densenet': dense_features
        }

    def get_layer(self, name):
        """Get specific layer for Grad-CAM visualization"""
        if name == 'darknet':
            return self.darknet.features[-1]  # Last conv layer of DarkNet53
        elif name == 'efficientnet':
            return self.efficientnet._conv_head  # Final conv layer of EfficientNet
        elif name == 'densenet':
            return self.densenet.features.denseblock4  # Final dense block
        else:
            raise ValueError(f"Unknown model name: {name}") 