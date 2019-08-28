
from datetime import datetime
import glob
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from .labels import labels
from .exceptions import *


# plants are indexed left to right, top to bottom
positions = [
    (1290, 670), (1730, 620), (2150, 590), (2580, 590),
    (3230, 630), (3615, 620), (4000, 640), (4470, 620),
    (1320, 1050), (1780, 990), (2150, 940), (2560, 910),
    (3270, 1070), (3660, 1060), (4045, 1080), (4450, 1080),
    (1367, 1419), (1794, 1380), (2162, 1367), (2583, 1346),
    (3281, 1404), (3654, 1452), (4053, 1431), (4449, 1436),
    (1389, 1823), (1793, 1803), (2195, 1767), (2580, 1776),
    (3294, 1805), (3680, 1802), (4086, 1778), (4457, 1803),
    (1397, 2211), (1794, 2199), (2189, 2189), (2639, 2205),
    (3303, 2201), (3675, 2159), (4064, 2147), (4467, 2177),
    (1386, 2582), (1821, 2588), (2219, 2597), (2642, 2607),
    (3303, 2588), (3665, 2615), (4062, 2574), (4463, 2547)
]

img_size = (5472, 3648)


class VIR(data.Dataset):
    """
    An abstract class. The parent class of all VIRs classes.
    """

    def __init__(self, root_dir: str, img_len: int, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        """
        :param root_dir: path to the Exp0 directory
        :param img_len: the length of the images in the dataset
        :param split_cycle: amount of days the data will be split by
        :param transform: optional transform to be applied on each frame
        """
        if max_len is None:
            max_len = 10000

        self.root_dir = root_dir
        self.vir_dirs = sorted(glob.glob(root_dir + '/*VIR_day'))
        self.vir_dirs = self._filter_dirs(self.vir_dirs, start_date, end_date)

        self.img_len = img_len
        self.split_cycle = split_cycle
        self.max_len = max_len

        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform

        # the type of the VIR images
        # to be assigned by inheriting classes
        self.vir_type = None

    def _filter_dirs(self, dirs, start_date, end_date):
        format = f"{self.root_dir}/%Y_%m_%d_%H_%M_%S_VIR_day"

        filtered = []
        for dir in dirs:
            date = datetime.strptime(dir, format)

            if start_date <= date <= end_date:
                filtered.append(dir)

        return filtered

    def __len__(self):
        return len(positions) * self.split_cycle

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError()

        # the day in the cycle this sample belongs to
        cycle_day = idx // len(positions)
        plant = idx % len(positions)

        tensors = []
        cur_day = self._get_day(self.vir_dirs[0])

        for vir_dir in self.vir_dirs:
            # update the current day when it changes
            if cur_day != self._get_day(vir_dir):
                cur_day = self._get_day(vir_dir)
                cycle_day -= 1

            # get the image only every split_cycle days
            if not cycle_day % self.split_cycle == 0:
                continue

            try:
                arr = self._get_np_arr(vir_dir, plant)
                tensor = torch.from_numpy(arr).float()
                tensor.unsqueeze_(0)
                tensors.append(self.transform(tensor))
            except DirEmptyError:
                pass

        image = torch.cat(tensors[:self.max_len])

        sample = {'image': image, 'label': labels[plant],
                  'position': positions[plant], 'plant': plant}

        return sample

    def _get_np_arr(self, vir_dir, plant_idx):
        pos = positions[plant_idx]

        left = pos[0] - self.img_len//2
        right = pos[0] + self.img_len//2
        top = pos[1] - self.img_len//2
        bottom = pos[1] + self.img_len//2

        image_path = glob.glob(f"{vir_dir}/*{self.vir_type}*.raw")
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]

        arr = np.fromfile(image_path, dtype=np.float16).reshape(3648, 5472)
        arr = arr[top:bottom, left:right]

        return arr

    # returns the date (day) of the directory
    def _get_day(self, lwir_dir):
        lwir_dir = lwir_dir[len(self.root_dir) + 1:]
        return lwir_dir.split('_')[2]


class VIR577nm(VIR):
    def __init__(self, root_dir: str, img_len=448, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        super().__init__(root_dir, img_len, split_cycle, start_date, end_date, max_len, transform)

        self.vir_type = "577nm"


class VIR692nm(VIR):
    def __init__(self, root_dir: str, img_len=448, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        super().__init__(root_dir, img_len, split_cycle, start_date, end_date, max_len, transform)

        self.vir_type = "692nm"


class VIR732nm(VIR):
    def __init__(self, root_dir: str, img_len=448, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        super().__init__(root_dir, img_len, split_cycle, start_date, end_date, max_len, transform)

        self.vir_type = "732nm"


class VIR970nm(VIR):
    def __init__(self, root_dir: str, img_len=448, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        super().__init__(root_dir, img_len, split_cycle, start_date, end_date, max_len, transform)

        self.vir_type = "970nm"


class VIRPolar(VIR):
    def __init__(self, root_dir: str, img_len=448, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        super().__init__(root_dir, img_len, split_cycle, start_date, end_date, max_len, transform)

        self.vir_type = "Polarizer"
