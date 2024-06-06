from dataloader import *
from model import *

path = '../model/'
transform = [transforms.Resize((SIZE, SIZE), Image.BICUBIC)]
train_loader, test_loader, val_loader = setup_dataloaders(transform=transform, load=False, shuffle=True, batch_size=2)
    
gen = get_generator(cin=1, cout=2, size=256)
opt = optim.Adam(gen.parameters(), lr=1e-4)
criterion = nn.L1Loss()        
pretrain_generator(gen, train_loader, opt, criterion, 20, verbose=1, path='../model/gen.pt')
model = ImColModel(1e-3, 1e-3, lam=100, gen=gen, load_pretrain_path='../model/gen.pt')

for L, ab in train_loader:
    inp = torch.cat([L, ab], dim=1)[0]
    test_img(model, inp)
    break
