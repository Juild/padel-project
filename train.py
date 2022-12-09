from torch import Tensor
import sys
sys.path.append("..")
from model import config
from model.model import BallClassifier
from model import dataset as ds
from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
from utils import import_data 
from torchvision.models import resnet50
from torchvision.ops import generalized_box_iou_loss
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

def train_model(train_loader, loss_func, learning_rate, epochs):
    resnet = resnet50(weights='DEFAULT')
    for param in resnet.parameters():
        param.requires_grad = False
    model = BallClassifier(base_model=resnet, num_classes=2).to(config.DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    # Number of "virtual batches"
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        model.train()
        loss = 0
        for (images, labels) in train_loader:
            images, labels = images.to(config.DEVICE, dtype=torch.float), labels.type(torch.LongTensor).to(config.DEVICE)
            opt.zero_grad()
            predicted_labels: Tensor = model(images)
            total_loss: Tensor = loss_func(predicted_labels, labels)
            loss += total_loss.item()
            print(f'Loss: {total_loss}')
            total_loss.backward()
            opt.step()
        train_loss.append(loss)

    return model, train_loss

def  evaluate_model(model, loss_func, data_loader, device):
    print(f'Evaluating model...')
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        total = 0
        correct = 0
        for images, labels in data_loader:
            images, labels = images.to(device, dtype=torch.float), labels.type(torch.LongTensor).to(device)
            # forward pass
            predicted_labels: Tensor = model(images)
            _, predicted = torch.max(predicted_labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()   
            loss: Tensor = loss_func(predicted_labels, labels)
            eval_loss += loss
        print(f'Accuracy of the network on the test images: {100 * correct // total} %')

def save_model(model, path):
    print(f"Saving model at {path}")
    torch.save(model, path)



print(f'Using device: {config.DEVICE}')

images_with_ball, images_without_ball = import_data()
# Image net MEAN and STDS
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

print(f'Using device: {config.DEVICE}')
print(f'Creating Dataset')
# probability of each transform is 0.5 by default
transforms = transforms.Compose(
    [
        transforms.Normalize(MEAN, STD)
    ]
)
train_dataset = ds.ImageDataset(
    images_with_ball,
    images_without_ball,
    transforms=transforms
)
batches = 512
print(f'Creating dataloader')
train_loader = DataLoader(
    train_dataset,
    batch_size=batches,
    shuffle=True,
    num_workers=(os.cpu_count() - 2),
    pin_memory=config.PIN_MEMORY
    )

# x: features , y: targets
loss_func = torch.nn.CrossEntropyLoss()
# Training
epochs = 20
model, train_loss = train_model(
    train_loader,
    loss_func=loss_func,
    learning_rate=.001,
    epochs=epochs
    )

# Plot metrics
plt.style.use('dark_background')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.plot(list(range(epochs)),train_loss)
plt.savefig(f'./figures/training_loss_history_{epochs}_{batches}.png')

# Evaluation
evaluate_model(model, loss_func=loss_func, data_loader=train_loader, device=config.DEVICE)
# Model saving
save_model(model, './box_regressor.pth')































