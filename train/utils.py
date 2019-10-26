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
    parser.add_argument('--color_max_len', dest='color_max_len', type=int, nargs='?', const=6, default=None,
                        help="""The maximum number of images in a single color sample.
                                    If not used it is unlimited,
                                    and if used with no number (i.e using --color_max_len with no value)
                                    it will have a default of 6.""")
    parser.add_argument('--skip', '--lwir_skip', dest='lwir_skip', type=int, nargs='?', const=5, default=1,
                        help="""The maximum number of images in a single vir sample.
                            If not used it is 1, and if used with no number (i.e using --lwir_skip or --skip with no value)
                            it will have a default of 5.""")
    parser.add_argument('-d', '--num_days', dest='num_days', type=int, default=None,
                        help='The number of days we use from the experiment start, default is all of the days.')
    parser.add_argument('-e', '--experiment', dest='experiment', required=True, choices=['Exp0', 'Exp1', 'Exp2'],
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


def get_levels_kernel(history_len: int):
    # Effective history formula: 1 + 2*(kernel_size-1)*(2^num_levels-1)
    if history_len <= 7:
        # effective history: 7
        kernel_size = 2
        num_levels = 2
    elif 7 <= history_len <= 15:
        # effective history: 15
        kernel_size = 2
        num_levels = 3
    elif 15 <= history_len <= 57:
        # effective history: 57
        kernel_size = 5
        num_levels = 3
    elif 57 <= history_len <= 121:
        # effective history: 121
        kernel_size = 5
        num_levels = 4
    elif 121 <= history_len <= 249:
        # effective history: 249
        kernel_size = 5
        num_levels = 5
    elif 249 <= history_len <= 311:
        # effective history: 311
        kernel_size = 6
        num_levels = 5
    else:
        # effective history: 505
        kernel_size = 5
        num_levels = 6

    return num_levels, kernel_size
