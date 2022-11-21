# imports
from torch.nn import Identity, ReLU, Linear, Sequential, Sigmoid, Module

#TODO Define custom model
class BoxRegressor(Module):
    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model
        self.regressor = Sequential(
			Linear(base_model.fc.in_features, 512),
			ReLU(),
			Linear(512, 256),
			ReLU(),
			Linear(256, 128),
			ReLU(),
			Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 8),
            ReLU(),
            Linear(8,4),
			Sigmoid()
		)
        self.base_model.fc = Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.regressor(x)
        return x