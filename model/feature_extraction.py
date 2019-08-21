import torch
from torch import nn
from torchvision.models import inception_v3


def greyscale_to_RGB(image: torch.Tensor, add_channels_dim=False) -> torch.Tensor:
    if add_channels_dim:
        image = image.unsqueeze(-3)

    dims = [-1] * len(image.shape)
    dims[-3] = 3
    return image.expand(*dims)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.inception = inception_v3(pretrained=True, transform_input=True, aux_logits=True)
        self.inception.fc = Identity()
        self.inception.eval()

        for p in self.inception.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        # if the image is not RGB
        if len(x.shape) < 4 or len(x.shape) >= 4 and x.shape[-3] == 1:
            x = greyscale_to_RGB(x, add_channels_dim=len(x.shape) < 4)

        return self.inception(x)
