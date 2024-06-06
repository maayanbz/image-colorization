import torch
import numpy as np
import matplotlib.pyplot as plt
from kornia.color import lab_to_rgb
from skimage.color import rgb2lab, lab2rgb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# speed up conv. computation
torch.backends.cudnn.benchmark = True


def normalize_lab(L, ab):
    return (L / 50 - 1), ab / 128


def inv_normalize_lab(L, ab):
    return (L + 1) * 50, ab * 128


def plot_losses(train_loss, val_loss, path=None, show=True):
    fig = plt.figure()
    fig.suptitle('Loss over epochs')
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if path is not None:
        fig.savefig(path)
    if show:
        plt.show()


def split_indices(n_samples, test_frac=0.2, val_frac=0.1):
    # randomly split indices to train (70%), val (10%) and test (20%)
    val_indices = list(np.random.choice(n_samples, int(n_samples*val_frac)))
    marked_indices = np.zeros(n_samples, dtype=int)
    marked_indices[val_indices] = 1
    left_indices = [i for i in range(n_samples) if marked_indices[i] != 1]
    test_indices = (np.random.choice(left_indices, int(n_samples*test_frac)))
    marked_indices[test_indices] = 1
    train_indices = [i for i in range(n_samples) if marked_indices[i] != 1]
    return train_indices, val_indices, test_indices


def eval_model(model, loader):
    """
    :input: model, loss function, data loader
    :output: average loss
    """
    model.eval()
    total_loss = 0
    for inputs, labels in loader:
        outputs = model(inputs)
        loss = model.loss_gen(inputs, labels, outputs)
        total_loss += loss.item()
        del loss, outputs
    total_loss /= len(loader)
    return total_loss


def pretrain_generator(gen, train_loader, opt, loss_fn, n_epochs, path=None, verbose=0):
    for epoch in range(n_epochs):
        total_loss = 0
        for L, ab in train_loader:
            outputs = gen(L)
            loss = loss_fn(outputs, ab)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            del loss, outputs
        if verbose > 0:
            print(f"Epoch {epoch + 1}/{n_epochs}")
    if path is not None:
        torch.save(gen.state_dict(), path)


def train_model(model, train_loader, val_loader=None, n_epochs=100, print_rate=100, path=None):
    """
    this function does NOT train the model
    """
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(n_epochs):
        train_loss = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = model.backprop(inputs, labels, outputs)
            train_loss += loss
            del loss, outputs
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        if val_loader is not None:
            val_loss = eval_model(model, val_loader)
            model.train()
            val_losses.append(val_loss)
        if epoch % print_rate == 0:
            print(f'Epoch {epoch}: train loss = {train_loss}', end='')
            if val_loader is not None:
                print(f', val loss = {val_losses[-1]}')
        if path is not None:
            model.save(path)
    return train_losses, val_losses


def test_img(model, img, path=None):
    # taking (3, 256, 256) img in scaled lab
    model.gen.eval()
    L = img[None, [0], :, :]
    ab = img[None, [1, 2], :, :]
    out = model(L).detach()
    model.gen.train()

    L = (L + 1) * 50
    ab *= 128
    out *= 128

    out_imgs = lab2rgb(torch.cat([L, out], dim=1).permute(0, 2, 3, 1).cpu().numpy()[0])
    orig_imgs = lab2rgb(torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()[0])

    out_imgs *= 255
    orig_imgs *= 255

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(orig_imgs.astype(np.uint8))
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(out_imgs.astype(np.uint8))
    ax2.set_title('Model Output')
    ax2.axis('off')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


# def test_img(model, img):
#     """
#     :input: test img in lab
#     :return: output img in rgb
#     """
#     input = img[:, [0], :, :]
#     out = model(input).detach()
#     result = torch.cat([input, out], dim=1)
#     return lab_to_rgb(result)
