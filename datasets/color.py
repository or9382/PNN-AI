from datetime import datetime
import glob
from PIL import Image
from torchvision.transforms import ToTensor

from .exceptions import *
from .experiments import plant_positions
from .ModalityDataset import ModalityDataset


class Color(ModalityDataset):
    def __init__(self, root_dir: str, exp_name: str, img_len: int = 255, split_cycle=7,
                 start_date=datetime(2019, 6, 4), end_date=datetime(2019, 7, 7),
                 max_len=None, transform=None):
        super().__init__(root_dir, exp_name, 'D465_Colo', img_len, plant_positions[exp_name].color_positions,
                         split_cycle, start_date, end_date, 1, max_len, transform)

    def _get_image(self, directory, plant_position):
        left = plant_position[0] - self.img_len // 2
        right = plant_position[0] + self.img_len // 2
        top = plant_position[1] - self.img_len // 2
        bottom = plant_position[1] + self.img_len // 2

        image_path = glob.glob(directory + '/*.jpg')
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]

        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))

        to_tensor = ToTensor()

        return to_tensor(image).float()

    def _dir_has_file(self, directory) -> bool:
        return len(glob.glob(directory + '/*.jpg')) != 0
