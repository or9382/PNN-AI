from datetime import datetime
import glob
import torch
from torch.utils import data
from torchvision import transforms
from abc import abstractmethod
from typing import Tuple

from .labels import labels
from .exceptions import *
from .transformations import RandomPNNTransform


class ModalityDataset(data.Dataset):
    """
    The parent class for datasets from the experiment.
    """

    def __init__(self, root_dir: str, exp_name: str, directory_suffix: str, img_len: int,
                 positions: Tuple[Tuple[int, int], ...], split_cycle=7, start_date=datetime(2019, 6, 4),
                 end_date=datetime(2019, 7, 7), skip=1, max_len=None, transform=None):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param directory_suffix: the directory name suffix for the image type
        :param positions: the positions of the plants within the images
        :param img_len: the length that the images will be resized to
        :param split_cycle: amount of days the data will be split by
        :param skip: how many frames to skip between ones taken
        :param max_len: the max amount of images to use; if None - no limit
        :param transform: optional transform to be applied on each frame
        """

        self.root_dir = root_dir
        self.directory_suffix = directory_suffix
        self.dirs = sorted(glob.glob(f'{root_dir}/*{directory_suffix}'))
        self.dirs = self.__filter_dirs(self.dirs, start_date, end_date)[::skip]

        self.exp_name = exp_name
        self.positions = positions

        self.num_plants = len(positions)

        self.img_len = img_len
        self.split_cycle = split_cycle

        if max_len is None:
            self.max_len = len(self.dirs) // split_cycle
        else:
            self.max_len = min(max_len, len(self.dirs) // split_cycle)

        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform

    def __get_dir_date(self, directory):
        dir_format = f"{self.root_dir}/%Y_%m_%d_%H_%M_%S_{self.directory_suffix}"
        return datetime.strptime(directory, dir_format)

    def __filter_dirs(self, dirs, start_date, end_date):
        return [d for d in dirs if start_date <= self.__get_dir_date(d) <= end_date and self._dir_has_file(d)]

    def __len__(self):
        return self.num_plants * self.split_cycle

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        # the day in the cycle this sample belongs to
        cycle_day = idx // self.num_plants
        plant = idx % self.num_plants

        tensors = []
        curr_date = self.__get_dir_date(self.dirs[0]).date()
        curr_date_idx = 0

        for directory in self.dirs:
            # update the current day when it changes
            dir_date = self.__get_dir_date(directory).date()
            if curr_date != dir_date:
                curr_date = dir_date
                curr_date_idx += 1

            # get the image only every split_cycle days
            if curr_date_idx % self.split_cycle == cycle_day:
                try:
                    image = self._get_image(directory, self.positions[plant])
                    tensors.append(image)
                except DirEmptyError:
                    pass

                if self.max_len is not None and len(tensors) >= self.max_len:
                    break

        for t in self.transform.transforms:
            if isinstance(t, RandomPNNTransform):
                t.new_params()

        image = torch.stack([self.transform(tensor) for tensor in tensors])

        sample = {'image': image, 'label': labels[self.exp_name][plant],
                  'position': self.positions[plant], 'plant': plant}

        return sample

    @abstractmethod
    def _get_image(self, directory, plant_position):
        pass

    @abstractmethod
    def _dir_has_file(self, directory) -> bool:
        pass
