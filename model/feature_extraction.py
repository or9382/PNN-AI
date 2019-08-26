import torch
from torch import nn
from torchvision.models import inception_v3
from TCN.tcn import TemporalConvNet
from typing import Dict, Union


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
        :return: a batch of feature vectors for each image of size (NxTx128)
        """
        # if the image is greyscale convert it to RGB
        if len(x.shape) < 5 or len(x.shape) >= 5 and x.shape[-3] == 1:
            x = greyscale_to_RGB(x, add_channels_dim=len(x.shape) < 5)

        # if we got a batch of sequences we have to calculate each sequence separately
        return torch.stack([self.inception(s) for s in x])


class ModalityFeatureExtractor(TemporalConvNet):
    def __init__(self, num_levels: int = 2, num_hidden: int = 60, embedding_size: int = 128, kernel_size=2,
                 dropout=0.2):
        """

        :param num_levels: the number of TCN layers
        :param num_hidden: number of channels used in the hidden layers
        :param embedding_size: size of final feature vector
        :param kernel_size: kernel size, make sure that it matches the feature vector size
        :param dropout: dropout probability
        :return: a TemporalConvNet matching the inputted params
        """
        num_channels = [num_hidden] * (num_levels - 1) + [embedding_size]
        super().__init__(2048, num_channels, kernel_size, dropout)

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


class PlantFeatureExtractor(nn.Module):
    def __init__(self, *default_mods: str, embedding_size: int = 128, **param_mods: Dict[str, Union[int, float]]):
        super().__init__()

        # make sure that no modality is appears as both default and with params
        assert len(set(default_mods).intersection(param_mods.keys())) == 0

        # all modalities
        self.mods = list(default_mods) + list(param_mods.keys())

        # make sure that we are using ANY modalities
        assert len(self.mods) > 0

        # create a feature extractor for images
        self.image_feat_ext = ImageFeatureExtractor()

        # a dictionary for the feature extractors for each modality
        self.mod_extractors: Dict[str, nn.Module] = dict()

        # create a feature extractor with default params for each default modality
        for mod in default_mods:
            self.mod_extractors[mod] = ModalityFeatureExtractor()
            self.add_module(f'{mod}_feat_extractor', self.mod_extractors[mod])

        # create a feature extractor with the inputted params for each param modality
        for mod in param_mods.keys():
            self.mod_extractors[mod] = ModalityFeatureExtractor(**param_mods[mod])
            self.add_module(f'{mod}_feat_extractor', self.mod_extractors[mod])

        self.final_feat_extractor = nn.Sequential(nn.Linear(128 * len(self.mods), embedding_size), nn.ReLU())

    def forward(self, **x: torch.Tensor):
        """

        :param x: input of type mod=x_mod for each modality where each x_mod is of shape
                    (NxT_modxC_modxH_modxW_mod),
                    where the batch size N is the same over all mods and others are mod-specific
        :return: a feature vector of shape (Nxembedding_size)
        """
        # make sure that all of the extractor mods and only them are used
        assert set(self.mods) == set(x.keys())

        # extract features from each image
        x = {mod: self.image_feat_ext(x[mod]) for mod in self.mods}

        # extract the features for each mod using the corresponding feature extractor
        x = {mod: self.mod_extractors[mod](x[mod]) for mod in self.mods}

        # take the final feature vector from each sequence
        x = torch.cat([x[mod][:, -1, :] for mod in self.mods], dim=1)

        # use the final linear feat extractor on all of these vectors
        return self.final_feat_extractor(x)
