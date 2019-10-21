from torch.utils.data import DataLoader, Dataset
import argparse

from .ModalityDataset import ModalityDataset
from .modalities import mod_map
from .experiments import get_all_modalities, get_experiment_modalities, experiments_info


class ModsImagesDataset(Dataset):
    def __init__(self, ds: ModalityDataset):
        # We assume that ds was initialised with skip=1, transform=None, max_len=None
        self.ds = ds
        # make sure that each image of the plant is alone
        cycle_dirs_lists = [list(l) for l in self.ds.cycles_dirs]
        self.ds.cycles_dirs = tuple(tuple([directory]) for directory in sum(cycle_dirs_lists, []))
        self.ds.split_cycle = len(self.ds.cycles_dirs)
        self.ds.max_len = 1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        res = self.ds.__getitem__(idx)
        return res['image']


def get_mod_norms(ds: ModalityDataset):
    loader = DataLoader(
        ModsImagesDataset(ds),
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


def main(args):
    used_mods = get_experiment_modalities(args.experiment) if args.modality is None else [args.modality]

    curr_experiment = experiments_info[args.experiment]

    if args.experiment_path is None:
        experiment_path = args.experiment
    else:
        experiment_path = args.experiment_path

    for mod in used_mods:
        ds = mod_map[mod](root_dir=experiment_path, exp_name=args.experiment,
                          start_date=curr_experiment.start_date, end_date=curr_experiment.end_date)
        mean, std = get_mod_norms(ds)
        print(f'Modality {mod}:')
        print(f'mean - {mean}\tstd - {std}')


if __name__ == '__main__':
    mods = get_all_modalities()
    parser = argparse.ArgumentParser(description='Calculate the mean and std of the modalities.')
    parser.add_argument('-e', '--experiment', dest='experiment', required=True, choices=['Exp0', 'Exp1', 'Exp2'],
                        help='The experiment we want to use.')
    parser.add_argument('-p', '--experiment_path', dest='experiment_path', type=str, default=None,
                        help='The path to the experiment root directory.')
    parser.add_argument('-m', '--modality', dest='modality', type=str, choices=mods, default=None,
                        help=f"""The modality you want to use. By default calculates for all modalities.
                        Choices are: {mods}""")

    arguments = parser.parse_args()
    main(arguments)
