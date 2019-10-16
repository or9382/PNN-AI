import argparse


def add_experiment_dataset_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('-s', '--split_cycle', dest='split_cycle', type=int, default=7,
                        help="The number of samples that each plant in the dataset will be split into.")
    parser.add_argument('--lwir_max_len', dest='lwir_max_len', type=int, nargs='?', const=44, default=None,
                        help="""The maximum number of images in a single lwir sample.
                            If not used it is unlimited, and if used with no number (i.e using --lwir_max_len with no value)
                            it will have a default of 44.""")
    parser.add_argument('--vir_max_len', dest='vir_max_len', type=int, nargs='?', const=6, default=None,
                        help="""The maximum number of images in a single vir sample.
                                If not used it is unlimited,
                                and if used with no number (i.e using --vir_max_len with no value)
                                it will have a default of 6.""")
    parser.add_argument('--skip', '--lwir_skip', dest='lwir_skip', type=int, nargs='?', const=5, default=1,
                        help="""The maximum number of images in a single vir sample.
                            If not used it is 1, and if used with no number (i.e using --lwir_skip or --skip with no value)
                            it will have a default of 5.""")
    parser.add_argument('-e', '--experiment', dest='experiment', required=True, choices=['EXP0', 'EXP1', 'EXP2'],
                        help='The experiment we want to use.')
    parser.add_argument('-p', '--experiment_path', dest='experiment_path', type=str, default=None,
                        help='The path to the experiment root directory.')


def __get_name(exp_name: str, file_type: str, excluded_modalities=[]):
    if len(excluded_modalities) == 0:
        return f'{exp_name}_{file_type}_all'
    else:
        excluded_modalities.sort()
        return '_'.join([exp_name, file_type, 'no'] + excluded_modalities)


def get_checkpoint_name(exp_name: str, excluded_modalities=[]):
    return __get_name(exp_name, 'checkpoint', excluded_modalities)


def get_feature_file_name(exp_name: str, excluded_modalities=[]):
    return f"{__get_name(exp_name, 'features', excluded_modalities)}.csv"


def get_tsne_name(exp_name: str, excluded_modalities=[], pca=0):
    return __get_name(exp_name, f'TSNE_{pca}' if pca > 0 else 'TSNE', excluded_modalities)


def get_used_modalities(modalities, excluded_modalities=[]):
    return {mod: args for mod, args in modalities.items() if mod not in excluded_modalities}
