# imports
from torch.utils.data import Dataset
from torch import nn

#TODO Define custom model
class ObjectDetector(nn.Module):
    def __init__(self, base_model, num_classes) -> None:
        super().__init__()
        self.base_model = base_model
        self.regressor = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
            
        )
		# set the classifier of our base model to produce outputs
		# from the last convolution block
        self.base_model.fc = nn.Identity()
    def forward(self, x):
        return self.regressor(x)