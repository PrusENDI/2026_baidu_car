from __future__ import annotations
from paddle import nn


class CnnModel(nn.Layer):
    """Official 128x128 lane CNN with [speed_demand, kappa_action] output."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2D(32, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2D(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv2D(64, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2D(64, 128, 3), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2D(128, 128, 3), nn.ReLU(), nn.Dropout(0.1),
            nn.Flatten(), nn.Linear(512, 128), nn.LeakyReLU(),
            nn.Linear(128, 32), nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(32, 2),
        )
    def forward(self, inputs):
        return self.features(inputs)


def set_feature_trainable(model: CnnModel, trainable: bool) -> None:
    """Keep compatibility with the existing staged fine-tuning scripts."""
    if trainable:
        for parameter in model.parameters():
            parameter.stop_gradient = False
        return
    for layer in model.features:
        if isinstance(layer, nn.Conv2D):
            for parameter in layer.parameters():
                parameter.stop_gradient = True
