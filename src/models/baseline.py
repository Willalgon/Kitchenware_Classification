import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    # Enrique sugiere probar con menos dropout si sigue habiendo "dientes de sierra",
    # por defecto lo dejamos a 0.5, pero si no llega al 40%, bajamos a 0.3 o 0.2
    def __init__(self, num_classes=6, dropout_rate=0.40):
        super().__init__()
        # Bloque 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   

        # Eliminamos el Drop2d de las convolucionales (Lo ha sugerido enrique para aprender más)
        # Solo dejamos el Dropout normal para las capas densas finales.
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # Extracción de características sin Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

    # Lo dejamos en logits sin softmax ya que el CCE lo incluye]
        return x