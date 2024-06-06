from dataloader import *
from model import *

path = '../model/'
gen = get_generator(cin=1, cout=2, size=256)
model = ImColModel(1e-4, 1e-4, lam=100, gen=gen, load_pretrain_path='../model/gen.pt')

transform = [transforms.Resize((SIZE, SIZE), Image.BICUBIC)]
train_loader, test_loader, val_loader = setup_dataloaders(transform=transform, load=False, shuffle=False, batch_size=4)

for l, ab in train_loader:
    inp = torch.cat([l, ab], dim=1)[0]
    test_img(model, inp, path='../img/tmp_img.jpg')
    break

# model = init_model(model)  # FUCK
train_loss, val_loss = train_model(model, train_loader, val_loader, n_epochs=30, print_rate=5, path=path)
plot_losses(train_loss, val_loss, path='../img/tmp_img.jpg', show=False)
model.save(path)
