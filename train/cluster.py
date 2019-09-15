
import torch
from torch.utils import data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .train_loop import restore_checkpoint, feat_ext, dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_features(feat_ext=feat_ext):
    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=4)
    restore_checkpoint()

    df = pd.DataFrame()
    for batch in dataloader:

        for key in batch:
            batch[key] = batch[key].to(device)

        labels = batch['label']

        x = batch.copy()

        del x['label']
        del x['plant']

        features = feat_ext(**x).numpy()

        batch_df = pd.DataFrame(data=features)
        batch_df.loc[:, 'label'] = labels.numpy()

        df = df.append(batch_df)

    df.to_csv('features.csv')


def pca_features():
    df = pd.read_csv('features.csv')
    pca = PCA(n_components=50)

    labels = df.labels
    df.drop('label', axis=1, inplace=True)

    pca_results = pca.fit_transform(df.values)

    df = pd.DataFrame(pca_results)
    df.loc[:, 'label'] = labels

    df.to_csv('features.csv')


def plot_tsne():
    df = pd.read_csv('features.csv')
    tsne = TSNE(n_components=2)

    tsne_results = tsne.fit_transform(df.values)
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]

    fig = plt.figure()
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="y",
        palette=sns.color_palette("hls", 4),
        data=df,
        legend="full",
        alpha=0.3,
    )
    fig.savefig('clusters')
