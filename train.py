from torch import Tensor
import sys
sys.path.append("..")
from model import model
from model import dataset as ds
from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
from utils import import_data 
from torchvision.models import resnet50
import logging
from torchmetrics import JaccardIndex
from torchvision.ops import box_iou, generalized_box_iou_loss
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

logging.basicConfig(filename='./debug.log', filemode='w', level='DEBUG')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

sys.stdout.write(f'Using device: {DEVICE}\n')

images, bboxes, means, stds = import_data(
    annotations_path="./datasets/annotations_yolo/",
    images_path="./frames/",
    channels=3
)
print(images[0].size())
train_dataset = ds.ImageDataset(
    images,
    bboxes=bboxes,
    transforms=transforms.Normalize(means, stds)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=PIN_MEMORY
    )


#TODO Read: https://d2l.ai/chapter_computer-vision/anchor.html


resnet = resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

model = model.BoxRegressor(base_model=resnet).to(DEVICE)
mse_loss_func = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=.001)
train_loss = []
epochs = 100
for epoch in range(epochs):
    model.train()
    loss: float = 0
    for (images, bboxes) in train_loader:
        (images, bboxes) = (images.to(DEVICE, dtype=torch.float), bboxes.to(DEVICE, torch.float))
        predictions: Tensor = model(images)
        iou_loss: Tensor = box_iou(predictions, bboxes).sum()
        mse_loss: Tensor = mse_loss_func(predictions, bboxes)
        total_loss: Tensor = mse_loss + iou_loss
        print(total_loss)
        print(predictions)
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        loss += float(total_loss)
    train_loss.append(loss)
sns.lineplot(list(range(epochs)),train_loss)
plt.show()



