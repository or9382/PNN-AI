from datetime import datetime
from typing import NamedTuple, Dict, Tuple, List
import torchvision.transforms as T


class ExpInfo(NamedTuple):
    start_date: datetime
    end_date: datetime
    modalities_norms: Dict[str, Tuple[List[float], List[float]]]


def get_experiment_modalities(exp_info: ExpInfo, lwir_skip: int, lwir_max_len: int, vir_max_len: int,):
    modalities: Dict[str, Dict] = {
        'lwir': {
            'max_len': lwir_max_len, 'skip': lwir_skip, 'transform': T.Compose(
                [T.Normalize(*[[norm] for norm in exp_info.modalities_norms['lwir']]), T.ToPILImage(),
                 T.RandomCrop(lwir_max_len, (206, 206)), T.RandomHorizontalFlip(lwir_max_len),
                 T.RandomVerticalFlip(lwir_max_len), T.ToTensor()])
        }
    }

    modalities.update(
        {
            mod: {
                'max_len': vir_max_len, 'transform': T.Compose(
                    [T.Normalize(*[[norm] for norm in norms]), T.ToPILImage(),
                     T.RandomCrop(vir_max_len, (412, 412)), T.RandomHorizontalFlip(vir_max_len),
                     T.RandomVerticalFlip(vir_max_len), T.ToTensor()])
            } for mod, norms in exp_info.modalities_norms.items() if mod != 'lwir'
        }
    )

    return modalities


experiments_info: Dict[str, ExpInfo] = {
    'EXP0': ExpInfo(
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
}
