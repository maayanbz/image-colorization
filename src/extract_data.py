import sys
import urllib.request
import tarfile
import os
import shutil

dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
data_dir = "../data/"
img_dir = "../data/jpg/"


def load_data(url, path):
    if not os.path.exists(path):
        os.makedirs(path)
    cfp, _ = urllib.request.urlretrieve(url)
    with tarfile.open(cfp, 'r:gz') as tar:
        tar.extractall(path=path)


def clean_data(path):
    shutil.rmtree(path)

# first_image_path = os.path.join(image_dir, paths[5])
# img = Image.open(first_image_path)
# plt.imshow(img)
