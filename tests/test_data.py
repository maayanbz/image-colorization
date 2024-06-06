import sys
sys.path.insert(1, '../src')

from dataloader import *
from utils import *

import urllib.request
import tarfile
import os
import matplotlib.pyplot as plt

trans = transforms.Resize((SIZE, SIZE), Image.BICUBIC)
train_loader, _, _ = setup_dataloaders(transforms=trans, load=False, shuffle=True)

print(next(iter(train_loader)))

clean_data(img_dir)
