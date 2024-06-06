from dataloader import *
from model import *

path = '../model/'
gen = get_generator(cin=1, cout=2, size=256)
model = ImColModel(3e-4, 3e-4, lam=50, gen=gen, load_pretrain_path='../model/gen.pt')

model.load(path)

transform = [transforms.Resize((SIZE, SIZE), Image.BICUBIC)]
train_loader, test_loader, val_loader = setup_dataloaders(transform=transform, load=False, shuffle=False, batch_size=1)


for l, ab in train_loader:
    inp = torch.cat([l, ab], dim=1)[0]
    test_img(model, inp)
    break

for l, ab in val_loader:
    inp = torch.cat([l, ab], dim=1)[0]
    test_img(model, inp)
    break

for l, ab in test_loader:
    inp = torch.cat([l, ab], dim=1)[0]
    test_img(model, inp)
    # break
