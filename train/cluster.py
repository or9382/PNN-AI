import torch
from torch.utils import data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import mixture
from sklearn import metrics

from .train_loop import restore_checkpoint, feat_ext, dataset
from datasets.labels import classes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_clusters = 6


def extract_features(feat_ext=feat_ext):
    feat_ext.eval()
    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=4)
    restore_checkpoint()

    df = pd.DataFrame()
    for batch in dataloader:

        for key in batch:
            batch[key] = batch[key].to(device)

        labels = batch['label'].cpu().numpy()
        labels = list(map(lambda i: classes[i], labels))
        plants = batch['plant']

        x = batch.copy()

        del x['label']
        del x['plant']

        features = feat_ext(**x).cpu().detach().numpy()

        batch_df = pd.DataFrame(data=features)
        batch_df.loc[:, 'label'] = labels
        batch_df.loc[:, 'plant'] = plants

        df = df.append(batch_df)

    df.to_csv('features.csv', index=False)


def pca_features():
    df = pd.read_csv('features.csv')
    pca = PCA(n_components=50)

    labels = df['label']
    plants = batch['plant']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    pca_results = pca.fit_transform(df.values)

    df = pd.DataFrame(pca_results)
    df.loc[:, 'label'] = labels
    df.loc[:, 'plant'] = plants

    df.to_csv('features.csv', index=False)


def plot_tsne():
    df = pd.read_csv('features.csv')
    tsne = TSNE(n_components=2, verbose=True)

    labels = df['label']
    plants = batch['plant']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    tsne_results = tsne.fit_transform(df.values)
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    df['label'] = labels
    df['plant'] = plants

    tsne_df = pd.DataFrame(data=tsne_results)
    tsne_df['label'] = labels
    tsne_df['plant'] = plants

    fig = plt.figure()
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="label",
        style="plant",
        palette=sns.color_palette("hls", 6),
        data=df,
        legend="full",
        alpha=0.3,
    )

    fig.savefig('clusters')
    tsne_df.to_csv('tsne2d.csv', index=False)


def eval_cluster(labels_true, labels_pred):
    print(f"\tARI: {metrics.adjusted_rand_score(labels_true, labels_pred)}")
    print(f"\tAMI: {metrics.adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')}")
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    print(f"\tHomogeneity: {homogeneity}")
    print(f"\tCompleteness: {completeness}")
    print(f"\tV-measure: {v_measure}")


def cluster_comp():
    df = pd.read_csv('tsne2d.csv')

    labels = df['label']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    print("KMeans:")
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=100).fit(df.values)
    eval_cluster(labels, kmeans.labels_)

    print("Spectral:")
    spectrals = cluster.SpectralClustering(n_clusters=n_clusters, assign_labels='discretize').fit(df.values)
    eval_cluster(labels, spectrals.labels_)

    print("GMM:")
    gmms = mixture.GaussianMixture(n_components=n_clusters).fit_predict(df.values)
    eval_cluster(labels, gmms)
