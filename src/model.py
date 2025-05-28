import torch
import torch.nn as nn
from torchvision import models
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc2(torch.relu(self.fc1(x)))

class EfficientNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = models.efficientnet_v2_s()
        in_feat = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_feat, n_classes)

    def forward(self, x):
        return self.model(x)

    def mlflow_signature(self):
        return ModelSignature(
            inputs=Schema([TensorSpec(np.float32, (None, 3, 32, 32), name="image")]),
            outputs=Schema([TensorSpec(np.float32, (None, 10), name="classes")])
        )