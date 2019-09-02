
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

    # make sure that the inception model stays on eval
    def train(self, mode=True):
        return self

    def forward(self, x: torch.Tensor):
        """

        :param x: a batch of image sequences of either size (NxTxCxHxW) or the squeezed size (NxTxHxW)
        :return: a batch of feature vectors for each image of size (NxTx2048)
        """
        # if the image is greyscale convert it to RGB
        if len(x.shape) < 5 or len(x.shape) >= 5 and x.shape[-3] == 1:
            x = greyscale_to_RGB(x, add_channels_dim=len(x.shape) < 5)

        # if we got a batch of sequences we have to calculate each sequence separately
        N, T = x.shape[:2]
        return self.inception(x.view(-1, *x.shape[2:])).view(N, T, -1)


class ModalityFeatureExtractor(TemporalConvNet):
    def __init__(self, num_levels: int = 3, num_hidden: int = 600, embedding_size: int = 128, kernel_size=2,
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

        :param x: input tensor of size (NxTx2048),
                where N is the batch size, T is the sequence length and 2048 is the input embedding dim
        :return: tensor of size (N x T x embedding_size) where out[:,t,:] is the output given all values up to time t
        """

        # transpose each sequence so that we get the correct size for the TemporalConvNet
        x = torch.stack([m.t() for m in x])

        out = super().forward(x)

        # undo the previous transpose
        return torch.stack([m.t() for m in out])


class PlantFeatureExtractor(nn.Module):
    def __init__(self, *default_mods: str, embedding_size: int = 512, **param_mods: Dict[str, Union[int, float]]):
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
            self.add_module(f'TCN_{mod}_feat_extractor', self.mod_extractors[mod])

        # create a feature extractor with the inputted params for each param modality
        for mod in param_mods.keys():
            self.mod_extractors[mod] = ModalityFeatureExtractor(**param_mods[mod])
            self.add_module(f'TCN_{mod}_feat_extractor', self.mod_extractors[mod])

        self.final_feat_extractor = nn.Linear(128 * len(self.mods), embedding_size)

        self.device = None
        self.streams = {mod: None for mod in self.mods}

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).get_device()
            if self.device != device:
                self.device = device
                self.streams = {mod: torch.cuda.Stream(device=self.device) for mod in self.mods}
        elif self.device is not None:
            self.device = None
            self.streams = {mod: None for mod in self.mods}

        return self

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
        if self.device is None:
            img_feats = {mod: self.image_feat_ext(x[mod]) for mod in self.mods}
        else:
            img_feats = {}
            for mod in self.mods:
                with torch.cuda.stream(self.streams[mod]):
                    img_feats[mod] = self.image_feat_ext(x[mod])

        # extract the features for each mod using the corresponding feature extractor
        if self.device is None:
            mod_feats = {mod: self.mod_extractors[mod](img_feats[mod]) for mod in self.mods}
        else:
            mod_feats = {}
            for mod in self.mods:
                with torch.cuda.stream(self.streams[mod]):
                    mod_feats[mod] = self.mod_extractors[mod](img_feats[mod])
            for mod in self.mods:
                self.streams[mod].synchronize()

        # take the final feature vector from each sequence
        combined_features = torch.cat([mod_feats[mod][:, -1, :] for mod in self.mods], dim=1)

        # use the final linear feat extractor on all of these vectors
        return self.final_feat_extractor(combined_features)
