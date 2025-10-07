import numpy as np
import matplotlib.pyplot as plt
import os

from constants import *

#Functions that create masks/annotations on images
def show_mask(mask, ax, random_color: bool = False) -> None:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375) -> None:
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=COLOUR_POINT_OUTLINE, marker='*', s=marker_size, edgecolor=COLOUR_LINE, linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color=COLOUR_BOX_OUTLINE, marker='*', s=marker_size, edgecolor=COLOUR_LINE, linewidth=1.25)   
    
def show_box(box, ax) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=COLOUR_BOX_OUTLINE, facecolor=(0,0,0,0), lw=2))    

def create_directory(name: str) -> None:
    if not os.path.exists(name):
        os.makedirs(name)
        print(f'Directory {name} created.')
