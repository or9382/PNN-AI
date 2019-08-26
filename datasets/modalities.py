
from torch.utils import data

from . import LWIR, VIR577nm, VIR692nm, VIR732nm, VIR970nm, VIRPolar
from .labels import labels


mod_map = {
    'lwir': LWIR,
    '577nm': VIR577nm,
    '692nm': VIR692nm,
    '732nm': VIR732nm,
    '970nm': VIR970nm,
    'polar': VIRPolar
}


class Modalities(data.Dataset):
    """
    A dataset class that lets the user decides which modalities to use.
    """

    def __init__(self, root_dir: str, img_len: int, modalities=None, split_cycle=7, transform=None):
        """
        :param root_dir: path to the Exp0 directory
        :param img_len: the length of the images in the dataset
        :param modalities: the modalities of the dataset; default: all.
        :param split_cycle: amount of days the data will be split by
        :param transform: optional transform to be applied on a sample
        """
        if modalities is None:
            modalities = mod_map.keys()

        if len(modalities) == 0:
            raise ValueError('should contain at least one modality')

        self.modalities = {
            mod: mod_map[mod](root_dir, img_len, split_cycle) for mod in modalities
        }

        self.transform = transform

    def __len__(self):
        dataset = next(iter(self.modalities.values()))
        return len(dataset)

    def __getitem__(self, idx):
        sample = {
            mod: dataset[idx]['image'] for mod, dataset in self.modalities.items()
        }

        plant = idx % len(labels)

        sample['label'] = labels[plant]
        sample['plant'] = plant

        if self.transform:
            sample = self.transform(sample)

        return sample
