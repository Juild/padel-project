from torch import Tensor
import sys
sys.path.append("..")
from model import config
from model.model import BoxRegressor
from model import dataset as ds
from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
from utils import import_data 
from torchvision.models import resnet50, resnet18
from torchvision.ops import box_iou, generalized_box_iou_loss
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

def train_model(train_loader, loss_func, learning_rate, epochs, virtual_batches):
    resnet = resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    model = BoxRegressor(base_model=resnet).to(config.DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    # Number of "virtual batches"
    accum_itr = 0
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        model.train()
        loss: float = 0
        batch_idx = 0
        for (images, bboxes) in train_loader:
            images, bboxes = images.to(config.DEVICE, dtype=torch.float), bboxes.to(config.DEVICE, torch.float)
            predicted_bboxes: Tensor = model(images)
            total_loss: Tensor = loss_func(predicted_bboxes, bboxes)
            loss += float(total_loss)/virtual_batches
            accum_itr += 1
            if accum_itr % virtual_batches == 0 or accum_itr == len(train_loader ) * train_loader.batch_size:
                print(f'Loss: {float(total_loss)}')
                opt.zero_grad()
                total_loss.backward()
                opt.step()
        train_loss.append(loss)

    return model, train_loss

def  evaluate_model(model, loss_func, data_loader, device):
    loss_history = []
    print(f'Evaluating model...')
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        for images, bboxes in data_loader:
            images, target_bboxes = images.to(device, dtype=torch.float), bboxes.to(device, dtype=torch.float)
            # forward pass
            predicted_bboxes: Tensor = model(images)
            loss: Tensor = loss_func(predicted_bboxes, target_bboxes)
            eval_loss += float(loss)
        loss_history.append(eval_loss)

def save_model(model, path):
    print(f"Saving model at {path}")
    torch.save(model, path)



print(f'Using device: {config.DEVICE}')

images, bboxes, means, stds = import_data(
    annotations_path="./datasets/annotations_yolo/",
    images_path="./frames/",
    channels=3
)
print(f'Using device: {config.DEVICE}')
print(f'Creating Dataset')
train_dataset = ds.ImageDataset(
    images,
    bboxes=bboxes,
    transforms=transforms.Normalize(means, stds)
)

print(f'Creating dataloader')
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=config.PIN_MEMORY
    )

loss_func = lambda x, y: mse_loss(x, y) + box_iou(x, y).mean()
# Training
epochs = 40
batches = 32
model, train_loss = train_model(train_loader, loss_func=loss_func, learning_rate=.01, epochs=epochs, virtual_batches=batches)

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































