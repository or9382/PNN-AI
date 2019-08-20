
import glob
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import ToTensor

from .exceptions import DirEmptyError


# plants are indexed left to right, top to bottom
positions = [
    (1290, 670), (1730, 620), (2150, 590), (2580, 590),
    (3230, 630), (3615, 620), (4000, 640), (4470, 620),
    (1320, 1050), (1780, 990), (2150, 940), (2560, 910),
    (3270, 1070), (3660, 1060), (40450, 1080), (4450, 1080),
    (1367, 1419), (1794, 1380), (2162, 1367), (2583, 1346),
    (3281, 1404), (3654, 1452), (4053, 1431), (4449, 1436),
    (1389, 1823), (1793, 1803), (2195, 1767), (2580, 1776),
    (3294, 1805), (3680, 1802), (4086, 1778), (4457, 1803),
    (1397, 2211), (1794, 2199), (2189, 2189), (2639, 2205),
    (3303, 2201), (3675, 2159), (4064, 2147), (4467, 2177),
    (1386, 2582), (1821, 2588), (2219, 2597), (2642, 2607),
    (3303, 2588), (3665, 2615), (4062, 2574), (4463, 2547)
]


class VIR(data.Dataset):
    """
    An abstract class. The parent class of all VIRs classes.
    """

    def __init__(self, root_dir: str, img_len: int, transform=None):
        """
        :param root_dir: path to the Exp1 directory
        :param img_len: the length of the images in the dataset
        :param transform: optional transform to be applied on a sample
        """
        self.vir_dirs = sorted(glob.glob(root_dir + '/*VIR_day'))
        self.img_len = img_len
        self.transform = transform

        # the type of the VIR images
        # to be assigned by inheriting classes
        self.vir_type = None

    def __len__(self):
        return len(positions)

    def __getitem__(self, idx):
        tensors = []
        to_tensor = ToTensor()

        for vir_dir in self.vir_dirs:
            try:
                image = self._get_image(vir_dir, idx)
                tensors.append(to_tensor(image))
            except DirEmptyError:
                pass

        return torch.cat(tensors)

    def _get_image(self, vir_dir, plant_idx):
        pos = positions[plant_idx]

        left = pos[0] - self.img_len//2
        right = pos[0] + self.img_len//2
        top = pos[1] - self.img_len//2
        bottom = pos[1] + self.img_len//2

        image_path = glob.glob(f"{vir_dir}/*{self.vir_type}*.raw")
        if len(image_path) == 0:
            raise DirEmptyError()

        image_path = image_path[0]

        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))

        return image


class VIR577nm(VIR):
    def __init__(self, root_dir: str, img_len: int, transform=None):
        super().__init__(root_dir, img_len, transform)

        self.vir_type = "577nm"


class VIR692nm(VIR):
    def __init__(self, root_dir: str, img_len: int, transform=None):
        super().__init__(root_dir, img_len, transform)

        self.vir_type = "692nm"


class VIR732nm(VIR):
    def __init__(self, root_dir: str, img_len: int, transform=None):
        super().__init__(root_dir, img_len, transform)

        self.vir_type = "732nm"


class VIR970nm(VIR):
    def __init__(self, root_dir: str, img_len: int, transform=None):
        super().__init__(root_dir, img_len, transform)

        self.vir_type = "970nm"


class VIRPolar(VIR):
    def __init__(self, root_dir: str, img_len: int, transform=None):
        super().__init__(root_dir, img_len, transform)

        self.vir_type = "Polarizer"
