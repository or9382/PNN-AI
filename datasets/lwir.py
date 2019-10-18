
from datetime import datetime
import glob
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

from .labels import labels
from .exceptions import *
from .experiments import plant_positions as positions
from .transformations import RandomPNNTransform


class LWIR(data.Dataset):
    """
    The LWIR data from the experiment.
    """
    def __init__(self, root_dir: str, exp_name: str, img_len=229, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 skip=1, max_len=None, transform=None):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param img_len: the length that the images will be resized to
        :param split_cycle: amount of days the data will be split by
        :param skip: how many frames to skip between ones taken
        :param max_len: the max amount of images to use; if None - no limit
        :param transform: optional transform to be applied on each frame
        """
        if max_len is None:
            max_len = 10000

        self.root_dir = root_dir
        self.lwir_dirs = sorted(glob.glob(root_dir + '/*LWIR'))[::skip]
        self.lwir_dirs = self._filter_dirs(self.lwir_dirs, start_date, end_date)

        self.exp_name = exp_name
        self.num_plants = len(positions[exp_name].lwir_positions)

        self.plant_crop_len = 60
        self.out_len = img_len
        self.split_cycle = split_cycle
        self.max_len = max_len

        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform

    def _filter_dirs(self, dirs, start_date, end_date):
        format = f"{self.root_dir}/%Y_%m_%d_%H_%M_%S_LWIR"

        filtered = []
        for dir in dirs:
            date = datetime.strptime(dir, format)

            if start_date <= date <= end_date:
                filtered.append(dir)

        return filtered

    def __len__(self):
        return self.num_plants * self.split_cycle

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        # the day in the cycle this sample belongs to
        cycle_day = idx // self.num_plants
        plant = idx % self.num_plants

        to_tensor = transforms.ToTensor()

        tensors = []
        cur_day = self._get_day(self.lwir_dirs[0])

        for lwir_dir in self.lwir_dirs:
            # update the current day when it changes
            if cur_day != self._get_day(lwir_dir):
                cur_day = self._get_day(lwir_dir)
                cycle_day -= 1

            # get the image only every split_cycle days
            if not cycle_day % self.split_cycle == 0:
                continue

            try:
                image = self._get_image(lwir_dir, plant)
                tensors.append(to_tensor(image).float())
            except DirEmptyError:
                pass

        for t in self.transform.transforms:
            if isinstance(t, RandomPNNTransform):
                t.new_params()

        tensors = tensors[:self.max_len]
        tensors = [self.transform(tensor) for tensor in tensors]
        image = torch.cat(tensors)

        sample = {'image': image, 'label': labels[self.exp_name][plant],
                  'position': positions[self.exp_name].lwir_positions[plant], 'plant': plant}

        return sample

    def _get_image(self, lwir_dir, plant_idx):
        pos = positions[self.exp_name].lwir_positions[plant_idx]

        left = pos[0] - self.plant_crop_len // 2
        right = pos[0] + self.plant_crop_len // 2
        top = pos[1] - self.plant_crop_len // 2
        bottom = pos[1] + self.plant_crop_len // 2

        image_path = glob.glob(lwir_dir + '/*.tiff')
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]

        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.out_len, self.out_len))

        return image

    # returns the date (day) of the directory
    def _get_day(self, lwir_dir):
        lwir_dir = lwir_dir[len(self.root_dir)+1:]
        return lwir_dir.split('_')[2]
