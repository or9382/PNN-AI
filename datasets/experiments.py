from datetime import datetime
from typing import NamedTuple, Dict, Tuple, List, Any
import torchvision.transforms as T
from .transformations import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, GreyscaleToRGB


class ExpPositions:
    def __init__(self, dictionary: Dict[str, Tuple[Tuple[int, int], ...]]):
        for key, value in dictionary.items():
            setattr(self, key, value)


class ExpInfo(NamedTuple):
    start_date: datetime
    end_date: datetime
    modalities_norms: Dict[str, Tuple[List[float], List[float]]]


def get_experiment_modalities_params(exp_info: ExpInfo, lwir_skip: int, lwir_max_len: int, vir_max_len: int):
    modalities: Dict[str, Dict] = {
        'lwir': {
            'max_len': lwir_max_len, 'skip': lwir_skip, 'transform': T.Compose(
                [T.Normalize(*exp_info.modalities_norms['lwir']), T.ToPILImage(),
                 RandomCrop((229, 229)), RandomHorizontalFlip(),
                 RandomVerticalFlip(), T.ToTensor(), GreyscaleToRGB()])
        }
    }

    modalities.update(
        {
            mod: {
                'max_len': vir_max_len, 'transform': T.Compose(
                    [T.Normalize(*norms), T.ToPILImage(),
                     RandomCrop((458, 458)), RandomHorizontalFlip(),
                     RandomVerticalFlip(), T.ToTensor(), GreyscaleToRGB()])
            } for mod, norms in exp_info.modalities_norms.items() if mod != 'lwir'
        }
    )

    return modalities


def get_all_modalities():
    return tuple(set(sum([list(info.modalities_norms.keys()) for info in experiments_info.values()], [])))


def get_experiment_modalities(exp_name: str):
    return list(experiments_info[exp_name].modalities_norms.keys())


experiments_info: Dict[str, ExpInfo] = {
    'Exp0': ExpInfo(
        datetime(2019, 6, 5),
        datetime(2019, 6, 19),
        {
            'lwir': ([21361.], [481.]),
            '577nm': ([.00607], [.00773]),
            '692nm': ([.02629], [.04364]),
            '732nm': ([.01072], [.11680]),
            '970nm': ([.00125], [.00095]),
            'polar': ([.05136], [.22331]),
        }
    ),
    'Exp1': ExpInfo(
        datetime(2019, 7, 28),
        datetime(2019, 8, 4),
        {
            'lwir': ([21458.6621], [119.2115]),
            '577nm': ([.0046], [.0043]),
            '692nm': ([.0181], [.0178]),
            '732nm': ([.0172], [.0794]),
            'polar_a': ([0.306], [.1435]),
        }
    ),
    'Exp2': ExpInfo(
        datetime(2019, 6, 5),
        datetime(2019, 6, 19),
        {
            'lwir': ([21361.], [481.]),
            '577nm': ([.00607], [.00773]),
            '692nm': ([.02629], [.04364]),
            '732nm': ([.01072], [.11680]),
            '970nm': ([.00125], [.00095]),
            'noFilter': ([.05136], [.22331]),
        }
    ),
}

# plants are indexed left to right, top to bottom
plant_positions = {
    'Exp0': ExpPositions({
        'lwir_positions': (
            (146, 105), (206, 100), (265, 97), (322, 98), (413, 105), (464, 105), (517, 110), (576, 115),
            (149, 157), (212, 152), (262, 145), (320, 142), (416, 167), (468, 165), (522, 169), (575, 171),
            (155, 207), (213, 205), (264, 204), (322, 200), (417, 213), (467, 218), (522, 216), (573, 219),
            (157, 263), (212, 261), (267, 258), (321, 260), (418, 266), (470, 266), (528, 263), (574, 270),
            (156, 317), (212, 315), (265, 315), (327, 319), (418, 321), (468, 314), (522, 314), (574, 319),
            (154, 366), (215, 368), (269, 372), (326, 374), (417, 373), (465, 375), (520, 373), (573, 369)
        ),
        'vir_positions': (
            (1290, 670), (1730, 620), (2150, 590), (2580, 590), (3230, 630), (3615, 620), (4000, 640), (4470, 620),
            (1320, 1050), (1780, 990), (2150, 940), (2560, 910), (3270, 1070), (3660, 1060), (4045, 1080), (4450, 1080),
            (1367, 1419), (1794, 1380), (2162, 1367), (2583, 1346), (3281, 1404), (3654, 1452), (4053, 1431), (4449, 1436),
            (1389, 1823), (1793, 1803), (2195, 1767), (2580, 1776), (3294, 1805), (3680, 1802), (4086, 1778), (4457, 1803),
            (1397, 2211), (1794, 2199), (2189, 2189), (2639, 2205), (3303, 2201), (3675, 2159), (4064, 2147), (4467, 2177),
            (1386, 2582), (1821, 2588), (2219, 2597), (2642, 2607), (3303, 2588), (3665, 2615), (4062, 2574), (4463, 2547)
        )
    }),
    'Exp1': ExpPositions({
        'lwir_positions': (
            (74, 64), (147, 64), (213, 72), (282, 64), (408, 70), (478, 72), (530, 77), (592, 78),
            (50, 119), (148, 132), (203, 123), (288, 144), (410, 132), (481, 141), (541, 144), (604, 135),
            (57, 196), (131, 197), (196, 211), (287, 211), (419, 200), (491, 206), (547, 207), (609, 206),
            (44, 263), (137, 258), (203, 274), (293, 271), (425, 269), (488, 267), (554, 264), (610, 279),
            (61, 333), (128, 329), (207, 329), (287, 333), (426, 330), (551, 331), (554, 330), (602, 336),
            (44, 393), (129, 389), (206, 391), (290, 390), (420, 391), (496, 410), (551, 403), (610, 393),
            (62, 460), (132, 456), (203, 454), (275, 470), (410, 463), (482, 473), (548, 465), (610, 456)
        ),
        'vir_positions': (
            (640, 270), (1292, 260), (1705, 275), (2287, 276), (3021, 340), (3673, 306), (4081, 270), (4590, 265),
            (560, 724), (1200, 815), (1700, 828), (2275, 842), (3132, 723), (3731, 764), (4168, 818), (4650, 804),
            (533, 1312), (1053, 1231), (1625, 1425), (2250, 1300), (3152, 1300), (3730, 1265), (4230, 1310), (4780, 1255),
            (500, 1765), (1200, 1800), (1716, 1860), (2350, 1786), (3205, 1780), (3725, 1730), (4245, 1690), (4720, 1730),
            (696, 2319), (1183, 2302), (1707, 2268), (2360, 2312), (3176, 2266), (3819, 2173), (4299, 2125), (4780, 2223),
            (445, 2823), (1200, 1524), (1643, 2406), (2381, 2350), (3137, 2273), (3825, 2186), (4320, 2155), (4802, 2228),
            (669, 3342), (1237, 3287), (1769, 3326), (2334, 3338), (3180, 3294), (3762, 3269), (4272, 3249), (4855, 3186)
        )
    }),
    'Exp2': ExpPositions({
        'lwir_positions': (
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        ),
        'vir_positions': (
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        )
    })
}
