def __get_name(file_type: str, excluded_modalities=[]):
    if len(excluded_modalities) == 0:
        return f'{file_type}_all'
    else:
        excluded_modalities.sort()
        return '_'.join([file_type, 'no'] + excluded_modalities)


def get_checkpoint_name(excluded_modalities=[]):
    return __get_name('checkpoint', excluded_modalities)


def get_feature_file_name(excluded_modalities=[]):
    return f"{__get_name('features', excluded_modalities)}.csv"


def get_tsne_name(excluded_modalities=[], pca=0):
    return __get_name(f'TSNE_{pca}' if pca > 0 else 'TSNE', excluded_modalities)


def get_used_modalities(modalities, excluded_modalities=[]):
    return {mod: args for mod, args in modalities.items() if mod not in excluded_modalities}
