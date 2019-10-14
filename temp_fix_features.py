import pandas as pd
from torch.utils import data
from math import ceil

from train.train_loop import dataset
from datasets import labels


def add_plant_ids():
    df = pd.read_csv('features.csv')

    df.to_csv('features_backup.csv', index=False)

    n_samples = len(df)
    n_plants = len(labels)
    plants_info = pd.Series(
        data=sum([list(range(n_plants)) for _ in range(ceil(n_samples / n_plants))], [])[:n_samples])

    df['plant'] = plants_info

    df.to_csv('features.csv', index=False)


if __name__ == '__main__':
    add_plant_ids()
