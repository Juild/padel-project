# imports
from torch.nn import Identity, ReLU, Linear, Sequential, Sigmoid, Module, Dropout, Softmax

#TODO Define custom model
class BallClassifier(Module):
    def __init__(self, base_model, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.base_model = base_model
        self.classifier = Sequential(
			Linear(base_model.fc.in_features, self.num_classes),

		)
        self.base_model.fc = Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x