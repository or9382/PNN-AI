
import glob
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import ToTensor

from .exceptions import *


# plants are indexed left to right, top to bottom
positions = [
    (146, 105), (206, 100), (265, 97), (322, 98),
    (413, 105), (464, 105), (517, 110), (576, 115),
    (149, 157), (212, 152), (262, 145), (320, 142),
    (416, 167), (468, 165), (522, 169), (575, 171),
    (155, 207), (213, 205), (264, 204), (322, 200),
    (417, 213), (467, 218), (522, 216), (573, 219),
    (157, 263), (212, 261), (267, 258), (321, 260),
    (418, 266), (470, 266), (528, 263), (574, 270),
    (156, 317), (212, 315), (265, 315), (327, 319),
    (418, 321), (468, 314), (522, 314), (574, 319),
    (154, 366), (215, 368), (269, 372), (326, 374),
    (417, 373), (465, 375), (520, 373), (573, 369)
]


class LWIR(data.Dataset):
    """
    The LWIR data from Exp0.
    """

    def __init__(self, root_dir: str, img_len=224, transform=None):
        """
        :param root_dir: path to the Exp1 directory
        :param img_len: the length that the images will be resized to
        :param transform: optional transform to be applied on a sample
        """
        self.lwir_dirs = sorted(glob.glob(root_dir + '/*LWIR'))
        self.plant_crop_len = 70
        self.out_len = img_len
        self.transform = transform

    def __len__(self):
        return len(positions)

    def __getitem__(self, idx):
        tensors = []
        to_tensor = ToTensor()

        for lwir_dir in self.lwir_dirs:
            try:
                image = self._get_image(lwir_dir, idx)
                tensors.append(to_tensor(image))
            except DirEmptyError:
                pass

        image = torch.cat(tensors)

        sample = {'image': image, 'position': positions[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_image(self, lwir_dir, plant_idx):
        pos = positions[plant_idx]

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
