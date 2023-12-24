import torch.nn as nn
from torchvision.models import vgg19
import config
import torch.nn.functional as F


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(
            config.DEVICE)

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input = (input+1) / 2.0
        target = (target+1) / 2.0
        target = F.interpolate(target, size=input.size()[2:], mode='nearest')
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        print("VGG Input shape:", vgg_input_features.shape)
        print("VGG Target shape:", vgg_target_features.shape)
        mse_loss = nn.functional.mse_loss(
            vgg_input_features, vgg_target_features)
        print("mse", mse_loss.item())
        return mse_loss
