from torch import Tensor
from typing import List
import torch
import sys
from model import config
from utils import import_test_image
import cv2
import numpy as np

image: Tensor = import_test_image(image_path='./datasets/frames/frame_4008.jpg')

# (n_chunks, height, width, BGR)

model = torch.load('./model.pth')

predictions = []
energies = []
with torch.no_grad():
    for chunk in image:
        # Permute dimensions as Pytorch expects (C, H, W)
        chunk = chunk.permute(2, 0, 1)
        # Add the batch dimension as the model is expecting it in the leading dimension
        chunk = chunk.unsqueeze(0) 
        chunk = chunk.to(config.DEVICE)
        energy = model(chunk)
        energies.append(energy.cpu())
        # We are just interested in the index, as it is the one representing the label
        # i.e. if first index has higher energy, label predicted is "0"
        _ , label = torch.max(energy, 1)
        predictions.append( label.item() )
    

print(predictions)
print(sum(predictions))
possible_balls = [{'label': x, 'idx': i} for i, x in enumerate(predictions) if x == 1]
for d in possible_balls:
    # now let's add the energies
    # [0] because it is the index of the energy associated to the ball class
    d['energy'] = energies[d['idx']][0][1].item()

# we sort the list by the highest energy
possible_balls.sort(reverse=True, key=lambda x: x['energy'])
print(possible_balls)
print(predictions[395], energies[395])
for i in range(20):
    predicted_chunk = image[possible_balls[i]['idx']]
    predicted_chunk = predicted_chunk.numpy().astype(np.uint8)
    cv2.imwrite(f'./predictions/prediction_{i}.jpg', predicted_chunk)
# k = 0
# for chunk in image:
#     chunk = chunk.numpy().astype(np.uint8)
#     cv2.imwrite(f'./predictions/original_{k}.jpg', chunk)
#     k += 1







