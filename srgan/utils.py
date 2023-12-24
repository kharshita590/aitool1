import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1-alpha) + device
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty


def save_checkpoint(model, optimizer,
                    filename='/home/akhilesh/gen/gen.pth.tar'):
    print("=> saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    try:
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            return
    except EOFError:
        return


def load_checkpoint(model, optimizer, lr, checkpoint_file):
    print("=> loading checkpoint")
    try:
        checkpoint_file = '/home/akhilesh/gen/gen.pth.tar'
        checkpoint = torch.load(checkpoint_file,
                                map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    except EOFError:
        print("stop")


def plot_examples(gen, num_samples=5):
    gen.eval()
    for i in range(num_samples):
        with torch.no_grad():
            noise = torch.randn(1, 15, 12, 1).to(config.DEVICE)
            upscaled_img = gen(noise)
        save_image(upscaled_img * 0.5 + 0.5, f"saved/sample_{i}.png")
    gen.train()
