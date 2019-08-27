
from datetime import datetime
from torch.utils import data
from typing import Dict

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

    def __init__(self, root_dir: str, *mods: str, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 transform=None, **k_mods: Dict):
        """
        :param root_dir: path to the Exp0 directory
        :param mods: modalities to be in the dataset, initialized with default arguments
        :param split_cycle: amount of days the data will be split by
        :param transform: optional transform to be applied on a sample
        :param k_mods: modalities to be in the dataset, as dictionaries of initialization arguments
        """
        if len(mods) + len(k_mods) == 0:
            mods = mod_map.keys()

        self.modalities = dict()

        for mod in mods:
            self.modalities[mod] = mod_map[mod](root_dir=root_dir, split_cycle=split_cycle,
                                                start_date=start_date, end_date=end_date)

        for mod in k_mods:
            self.modalities[mod] = mod_map[mod](root_dir=root_dir, split_cycle=split_cycle,
                                                start_date=start_date, end_date=end_date, **k_mods[mod])

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
