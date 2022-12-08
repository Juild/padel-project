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

def train_model(train_loader, loss_func, learning_rate, epochs, virtual_batches):
    resnet = resnet50(weights='DEFAULT')
    for param in resnet.parameters():
        param.requires_grad = False
    model = BallClassifier(base_model=resnet, num_classes=2).to(config.DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    # Number of "virtual batches"
    accum_itr = 0
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        model.train()
        loss = 0
        batch_idx = 0
        for (images, labels) in train_loader:
            images, labels = images.to(config.DEVICE, dtype=torch.float), labels.type(torch.LongTensor).to(config.DEVICE)
            predicted_labels: Tensor = model(images)
            total_loss: Tensor = loss_func(predicted_labels, labels)
            loss += total_loss
            accum_itr += 1
            if accum_itr % virtual_batches == 0 or accum_itr == len(train_loader ) * train_loader.batch_size:
                print(f'Loss: {float(total_loss)}')
                total_loss.backward()
                opt.zero_grad()
                opt.step()
        train_loss.append(loss)

    return model, train_loss

def  evaluate_model(model, loss_func, data_loader, device):
    loss_history = []
    print(f'Evaluating model...')
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        for images, labels in data_loader:
            images, target_labels = images.to(device, dtype=torch.float), labels.type(torch.LongTensor).to(device)
            # forward pass
            predicted_labels: Tensor = model(images)
            for label in predicted_labels:
                 print(torch.max(predicted_labels))
            loss: Tensor = loss_func(predicted_labels, target_labels)
            eval_loss += loss
        loss_history.append(float(eval_loss))

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

print(f'Creating dataloader')
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=(os.cpu_count() - 2),
    pin_memory=config.PIN_MEMORY
    )

# x: features , y: targets
loss_func = torch.nn.CrossEntropyLoss()
# Training
epochs = 5
batches = 32
model, train_loss = train_model(train_loader, loss_func=loss_func, learning_rate=.001, epochs=epochs, virtual_batches=batches)

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































