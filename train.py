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

logging.basicConfig(filename='./debug.log', filemode='w', level='DEBUG')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    num_workers=1
    )


#TODO Read: https://d2l.ai/chapter_computer-vision/anchor.html


resnet = resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

model = model.BoxRegressor(base_model=resnet).to(DEVICE)
bbox_loss_func = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=.01)
train_loss = []
logging.info('before for loop')
for epoch in range(50):
    model.train()
    loss = 0
    logging.info('inside for loop 1')
    for (images, bboxes) in train_loader:
        logging.info('inside for loop')
        (images, bboxes) = (images.to(DEVICE, dtype=torch.float), bboxes.to(DEVICE, torch.float))
        predictions = model(images)
        bbox_loss = bbox_loss_func(predictions, bboxes)
        loss += bbox_loss
        opt.zero_grad()
        bbox_loss.backward()
        opt.step()
        print(loss)
        print("Real ", bboxes)
        print("Predicted ", predictions)