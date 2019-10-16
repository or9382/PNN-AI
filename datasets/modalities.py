
from datetime import datetime
from torch.utils import data
from typing import Dict, List
import random

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

    def __init__(self, root_dir: str, exp_name: str, *mods: str, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 transform=None, **k_mods: Dict):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param mods: modalities to be in the dataset, initialized with default arguments
        :param split_cycle: amount of days the data will be split by
        :param transform: optional transform to be applied on a sample
        :param k_mods: modalities to be in the dataset, as dictionaries of initialization arguments
        """
        if len(mods) + len(k_mods) == 0:
            mods = mod_map.keys()

        self.modalities = dict()

        for mod in mods:
            self.modalities[mod] = mod_map[mod](root_dir=root_dir, exp_name=exp_name, split_cycle=split_cycle,
                                                start_date=start_date, end_date=end_date)

        for mod in k_mods:
            self.modalities[mod] = mod_map[mod](root_dir=root_dir, exp_name=exp_name, split_cycle=split_cycle,
                                                start_date=start_date, end_date=end_date, **k_mods[mod])

        self.transform = transform

        self.exp_name = exp_name

        self.split_cycle = split_cycle

        self.num_plants = len(labels[exp_name])

    def __len__(self):
        dataset = next(iter(self.modalities.values()))
        return len(dataset)

    def __getitem__(self, idx):
        sample = {
            mod: dataset[idx]['image'] for mod, dataset in self.modalities.items()
        }

        plant = idx % self.num_plants

        sample['label'] = labels[self.exp_name][plant]
        sample['plant'] = plant

        if self.transform:
            sample = self.transform(sample)

        return sample


class ModalitiesSubset(data.Dataset):
    def __init__(self, modalities: Modalities, plants: List[int]):
        self.data = modalities
        self.split_cycle = modalities.split_cycle
        self.plants = plants
        self.num_plants = len(plants)

    def __len__(self):
        return self.num_plants * self.split_cycle

    def __getitem__(self, idx):
        plant = self.plants[idx % self.num_plants]
        cycle = idx // self.num_plants

        data = self.data[self.data.num_plants * cycle + plant]
        data['plant'] = idx % self.num_plants

        return data

    @staticmethod
    def random_split(modalities: Modalities, plants_amounts: List[int]):
        idx = list(range(modalities.num_plants))
        random.shuffle(idx)

        subsets = []
        for amount in plants_amounts:
            subsets.append(ModalitiesSubset(modalities, idx[:amount]))
            idx = idx[amount:]

        return subsets

    @staticmethod
    def leave_one_out(modalities: Modalities, plant_idx: int):
        rest_idx = list(range(modalities.num_plants))
        del rest_idx[plant_idx]

        one_out = ModalitiesSubset(modalities, [plant_idx])
        rest = ModalitiesSubset(modalities, rest_idx)

        return one_out, rest
