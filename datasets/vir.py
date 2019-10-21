from datetime import datetime
import glob
import numpy as np
import torch

from .exceptions import *
from .experiments import plant_positions
from .ModalityDataset import ModalityDataset


# img_size = (5472, 3648)


class VIR(ModalityDataset):
    """
    An abstract class. The parent class of all VIRs classes.
    """

    def __init__(self, root_dir: str, exp_name: str, vir_type: str, img_len: int = 510, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param vir_type: the name of the used vir filter
        :param img_len: the length of the images in the dataset
        :param split_cycle: amount of days the data will be split by
        :param transform: optional transform to be applied on each frame
        """
        super().__init__(root_dir, exp_name, 'VIR_day', img_len, plant_positions[exp_name].vir_positions, split_cycle,
                         start_date, end_date, 1, max_len, transform)
        self.vir_type = vir_type

    def _get_image(self, directory, plant_position):
        left = plant_position[0] - self.img_len // 2
        right = plant_position[0] + self.img_len // 2
        top = plant_position[1] - self.img_len // 2
        bottom = plant_position[1] + self.img_len // 2

        image_path = glob.glob(f"{directory}/*{self.vir_type}*.raw")
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]

        arr = np.fromfile(image_path, dtype=np.int16).reshape(*self.__get_image_dims(image_path))
        arr = arr[top:bottom, left:right].astype(np.float) / self.__get_exposure(image_path)

        return torch.from_numpy(arr).float().unsqueeze(0)

    @staticmethod
    def __get_exposure(file_name: str):
        return float(file_name.split('ET')[-1].split('.')[0])

    @staticmethod
    def __get_image_dims(file_name: str):
        fields = file_name.split('/')[-1].split('_')
        return int(fields[8]), int(fields[7])


class VIR577nm(VIR):
    def __init__(self, *args, **kwargs):
        super().__init__(vir_type="577nm", *args, **kwargs)


class VIR692nm(VIR):
    def __init__(self, *args, **kwargs):
        super().__init__(vir_type="692nm", *args, **kwargs)


class VIR732nm(VIR):
    def __init__(self, *args, **kwargs):
        super().__init__(vir_type="732nm", *args, **kwargs)


class VIR970nm(VIR):
    def __init__(self, *args, **kwargs):
        super().__init__(vir_type="970nm", *args, **kwargs)


class VIRPolar(VIR):
    def __init__(self, *args, **kwargs):
        super().__init__(vir_type="Polarizer", *args, **kwargs)


class VIRPolarA(VIR):
    def __init__(self, *args, **kwargs):
        super().__init__(vir_type="PolarizerA", *args, **kwargs)
