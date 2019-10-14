import pandas as pd
from torch.utils import data

from train.train_loop import dataset

if __name__ == '__main__':
    df = pd.read_csv('features.csv')

    df.to_csv('features_backup.csv', index=False)

    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=4)
    plants_info = pd.Series().append([pd.Series(data=batch['plant']) for batch in dataloader])

    df['plant'] = plants_info

    df.to_csv('features.csv', index=False)
