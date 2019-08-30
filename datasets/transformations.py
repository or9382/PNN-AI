
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class RandomBrightness:
    def __init__(self, len):
        self.len = len
        self.i = 0
        self.params = None

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = self.get_params()

        self.i = (self.i + 1) % self.len

        return TF.adjust_brightness(img, self.params)


class RandomContrast:
    def __init__(self, len):
        self.len = len
        self.i = 0
        self.params = None

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = self.get_params()

        self.i = (self.i + 1) % self.len

        return TF.adjust_contrast(img, self.params)


class RandomGamma:
    def __init__(self, len):
        self.len = len
        self.i = 0
        self.params = None

    @staticmethod
    def get_params():
        gamma = random.uniform(0.5, 2)
        gain = random.uniform(0.5, 2)

        return gamma, gain

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = self.get_params()

        self.i = (self.i + 1) % self.len

        return TF.adjust_contrast(img, *self.params)


class RandomHue:
    def __init__(self, len):
        self.len = len
        self.i = 0
        self.params = None

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = self.get_params()

        self.i = (self.i + 1) % self.len

        return TF.adjust_hue(img, self.params)


class RandomSaturation:
    def __init__(self, len):
        self.len = len
        self.i = 0
        self.params = None

    @staticmethod
    def get_params():
        return random.uniform(0.5, 2)

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = self.get_params()

        self.i = (self.i + 1) % self.len

        return TF.adjust_saturation(img, self.params)


class RandomCrop:
    def __init__(self, len, out_size):
        self.len = len
        self.i = 0
        self.params = None
        self.out_size = out_size

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = T.RandomCrop.get_params(img, self.out_size)

        self.i = (self.i + 1) % self.len

        return TF.crop(img, *self.params)


class RandomHorizontalFlip:
    def __init__(self, len, p=0.5):
        self.len = len
        self.i = 0
        self.params = None
        self.p = p

    def get_params(self):
        return random.random() < self.p

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = self.get_params()

        self.i = (self.i + 1) % self.len

        if self.params:
            return TF.hflip(img)

        return img


class RandomVerticalFlip:
    def __init__(self, len, p=0.5):
        self.len = len
        self.i = 0
        self.params = None
        self.p = p

    def get_params(self):
        return random.random() < self.p

    def __call__(self, img: Image.Image):
        if self.i == 0:
            self.params = self.get_params()

        self.i = (self.i + 1) % self.len

        if self.params:
            return TF.vflip(img)

        return img
