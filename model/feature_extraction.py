import torch
from torch import nn
from torchvision.models import inception_v3
from TCN.tcn import TemporalConvNet


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
        """

        :param x: a batch of image sequences of either size (NxTxCxHxW) or the squeezed size (NxTxHxW)
        :return:
        """
        # if the image is greyscale convert it to RGB
        if len(x.shape) < 5 or len(x.shape) >= 5 and x.shape[-3] == 1:
            x = greyscale_to_RGB(x, add_channels_dim=len(x.shape) < 5)

        # if we got a batch of sequences we have to calculate each sequence separately
        return torch.stack([self.inception(s) for s in x])


class ModalityFeatureExtractor(TemporalConvNet):
    def __init__(self, in_size: int, num_levels: int, num_hidden: int, embedding_size: int, kernel_size=2, dropout=0.2):
        """

        :param in_size: the size of each vector in the input sequence
        :param num_levels: the number of TCN layers
        :param num_hidden: number of channels used in the hidden layers
        :param embedding_size: size of final feature vector
        :param kernel_size: kernel size, make sure that it matches the feature vector size
        :param dropout: dropout probability
        :return: a TemporalConvNet matching the inputted params
        """
        num_channels = [num_hidden] * (num_levels - 1) + [embedding_size]
        super().__init__(in_size, num_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor):
        """

        :param x: input tensor of size (NxTxD),
                where N is the batch size, T is the sequence length and D is the input embedding dim
        :return: tensor of size (N x T x embedding_size) where out[:,t,:] is the output given all values up to time t
        """

        # transpose each sequence so that we get the correct size for the TemporalConvNet
        x = torch.stack([m.t() for m in x])

        out = super().forward(x)

        # undo the previous transpose
        return torch.stack([m.t() for m in out])
