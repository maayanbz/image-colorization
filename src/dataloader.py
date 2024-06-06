from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from extract_data import *
import torch
from utils import *
import sys

from kornia.color import rgb_to_lab, lab_to_rgb

SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, image_dir, data, name=None, transform=None):
        super().__init__()
        self.transform = transforms.Compose(transform) if transform is not None else None
        self.size = SIZE
        self.image_dir = image_dir
        self.data = data.to(device)
        self.name = name
        self._transform_data()
        print(f'{self.name}: {torch.numel(self.data)*self.data.element_size()//1e6}MB in GPU')

    def _transform_data(self):
        if self.transform is not None:
            self.data = self.transform(self.data)
        # self.data /= 255

    def __getitem__(self, idx):
        img = self.data[idx].float()
        img /= 255
        img = rgb_to_lab(img)
        L, ab = img[[0], :, :], img[[1, 2], :, :]
        L, ab = normalize_lab(L, ab)
        return L, ab

    def __len__(self):
        return self.data.shape[0]


def get_all_data(image_dir, paths, transform):
    if transform is None:
        transform = []
    data = []
    transform = transforms.Compose(transform + [transforms.PILToTensor()])
    for path in paths:
        path = os.path.join(image_dir, path)
        img = Image.open(path).convert("RGB")
        data.append(transform(img))
    return torch.stack(data, dim=0)


def setup_dataloaders(batch_size=128, transform=None, load=False, shuffle=False):
    if load:
        load_data(dataset_url, data_dir)
    n_samples = len(os.listdir(img_dir))
    # n_samples = 2000 # !!!
    if shuffle:
        train_ind, test_ind, val_ind = split_indices(n_samples)  # 70-20-10 split
        torch.save(train_ind, data_dir + 'split/train.pt')
        torch.save(test_ind, data_dir + 'split/test.pt')
        torch.save(val_ind, data_dir + 'split/val.pt')
    train_ind = torch.load(data_dir + 'split/train.pt')
    test_ind = torch.load(data_dir + 'split/test.pt')
    val_ind = torch.load(data_dir + 'split/val.pt')

    paths = os.listdir(img_dir)#[:2000] # !!!
    data = get_all_data(img_dir, paths, transform)

    # train_paths = [paths[i] for i in train_ind]
    # test_paths = [paths[i] for i in test_ind]
    # val_paths = [paths[i] for i in val_ind]

    trainset = ColorizationDataset(img_dir, data[train_ind], name='train', transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size)

    testset = ColorizationDataset(img_dir, data[test_ind], name='test', transform=transform)
    test_loader = DataLoader(testset, batch_size=1)

    valset = ColorizationDataset(img_dir, data[val_ind], name='val', transform=transform)
    val_loader = DataLoader(valset, batch_size=batch_size)

    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    transform = [transforms.Resize((SIZE, SIZE), Image.BICUBIC)]
    train_loader, _, _ = setup_dataloaders(transform=transform, shuffle=True, batch_size=1)

    for L, ab in train_loader:
        L, ab = inv_normalize_lab(L, ab)
        img = torch.cat([L, ab], dim=1)
        img = lab_to_rgb(img)
        img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
        # img = lab_to_rgb(torch.cat([L, ab], dim=1)).permute(0, 2, 3, 1).cpu().numpy()[0]
        img *= 255
        plt.imshow(img.astype(np.uint8))
        plt.show()
        break

