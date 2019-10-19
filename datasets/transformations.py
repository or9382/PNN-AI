from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from abc import abstractmethod


class RandomPNNTransform:
    def __init__(self):
        self.params = None

    @staticmethod
    @abstractmethod
    def get_params():
        pass

    def new_params(self):
        self.params = self.get_params()


class RandomBrightness(RandomPNNTransform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        return TF.adjust_brightness(img, self.params)


class RandomContrast(RandomPNNTransform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        return TF.adjust_contrast(img, self.params)


class RandomGamma(RandomPNNTransform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params():
        gamma = random.uniform(0.5, 2)
        gain = random.uniform(0.5, 2)

        return gamma, gain

    def __call__(self, img: Image.Image):
        return TF.adjust_contrast(img, *self.params)


class RandomHue(RandomPNNTransform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        return TF.adjust_hue(img, self.params)


class RandomSaturation(RandomPNNTransform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        return TF.adjust_saturation(img, self.params)


class RandomCrop(RandomPNNTransform):
    def __init__(self, out_size):
        self.out_size = out_size
        super().__init__()

    @staticmethod
    def get_params():
        return None

    def __call__(self, img: Image.Image):
        if self.params is None:
            self.params = T.RandomCrop.get_params(img, self.out_size)

        return TF.crop(img, *self.params)


class RandomHorizontalFlip(RandomPNNTransform):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def get_params(self):
        return random.random() < self.p

    def __call__(self, img: Image.Image):
        if self.params:
            return TF.hflip(img)

        return img


class RandomVerticalFlip(RandomPNNTransform):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def get_params(self):
        return random.random() < self.p

    def __call__(self, img: Image.Image):
        if self.params:
            return TF.vflip(img)

        return img


class GreyscaleToRGB:
    def __call__(self, img: Image.Image):
        return img.convert('RGB')
