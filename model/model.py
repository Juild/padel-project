# imports
from torch.utils.data import Dataset
from torch import nn

#TODO Define custom model
class BoxRegressor(nn.Module):
    def __init__(self, base_model, num_classes) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=10),
            nn.ReLU(),
            nn.Flatten(end_dim=1),
            nn.Linear(in_features=3*3*1080*1920, out_features=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(x)