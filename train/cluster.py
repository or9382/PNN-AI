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
import argparse
from typing import List

from datasets.labels import classes
from datasets import Modalities
from datasets.experiments import experiments_info
from model.feature_extraction import PlantFeatureExtractor as FeatureExtractor
from .utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_extractor(experiment_name: str, modalities: List[str], excluded_modalities: List[str] = []):
    checkpoint_name = get_checkpoint_name(experiment_name, excluded_modalities)

    used_modalities = get_used_modalities(modalities, excluded_modalities)
    feat_extractor = FeatureExtractor(*used_modalities).to(device)

    checkpoint = torch.load(checkpoint_name)
    feat_extractor.load_state_dict(checkpoint['feat_ext_state_dict'])

    return feat_extractor.to(device)


def extract_features(modalities: List[str], split_cycle: int, start_date, end_date, experiment_name: str,
                     experiment_root_dir: str, excluded_modalities: List[str] = []):
    used_modalities = get_used_modalities(modalities, excluded_modalities)
    dataset = Modalities(experiment_root_dir, experiment_name, split_cycle=split_cycle, start_date=start_date,
                         end_date=end_date, **used_modalities)

    feat_extractor = load_extractor(experiment_name, excluded_modalities).eval()
    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=4)

    df = pd.DataFrame()
    for batch in dataloader:

        for key in batch:
            batch[key] = batch[key].to(device)

        labels = batch['label'].cpu().numpy()
        labels = list(map(lambda i: classes[i], labels))
        plants = batch['plant'].cpu().numpy()

        x = batch.copy()

        del x['label']
        del x['plant']

        features = feat_extractor(**x).cpu().detach().numpy()

        batch_df = pd.DataFrame(data=features)
        batch_df.loc[:, 'label'] = labels
        batch_df.loc[:, 'plant'] = plants

        df = df.append(batch_df, ignore_index=True)

    df.to_csv(get_feature_file_name(experiment_name, excluded_modalities), index=False)
    print('Finished extraction.')
    return df


def pca_features(df: pd.DataFrame, n_components=50):
    pca = PCA(n_components=n_components)

    labels = df['label']
    plants = df['plant']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    pca_results = pca.fit_transform(df.values)

    df = pd.DataFrame(pca_results)
    df.loc[:, 'label'] = labels
    df.loc[:, 'plant'] = plants

    return df


def plot_tsne(df: pd.DataFrame, experiment_name: str, excluded_modalities=[], pca=0):
    if pca > 0:
        df = pca_features(df, pca)
        print('Finished PCA.')

    tsne = TSNE(n_components=2, verbose=True)

    labels = df['label']
    plants = df['plant']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    tsne_results = tsne.fit_transform(df.values)
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    df['label'] = labels
    df['plant'] = plants

    tsne_df = pd.DataFrame(data=tsne_results)
    tsne_df.loc[:, 'label'] = labels

    fig = plt.figure(figsize=(25.6, 19.2))
    ax = sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="label",
        palette=sns.color_palette("hls", 6),
        data=df,
        legend="full",
        alpha=0.3,
        s=250,
    )

    for x, y, plant in zip(df['tsne-one'], df['tsne-two'], plants):
        ax.annotate(str(plant), (x, y), fontsize='large', ha="center")

    tsne_name = get_tsne_name(experiment_name, excluded_modalities, pca)

    fig.savefig(f'{tsne_name}_clusters', bbox_inches="tight")
    tsne_df.to_csv(f'{tsne_name}2d.csv', index=False)


def eval_cluster(labels_true, labels_pred):
    print(f"\tARI: {metrics.adjusted_rand_score(labels_true, labels_pred)}")
    print(f"\tAMI: {metrics.adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')}")
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    print(f"\tHomogeneity: {homogeneity}")
    print(f"\tCompleteness: {completeness}")
    print(f"\tV-measure: {v_measure}")


def cluster_comp(df: pd.DataFrame, num_clusters=6):
    labels = df['label']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    print("KMeans:")
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=100).fit(df.values)
    eval_cluster(labels, kmeans.labels_)

    print("Spectral:")
    spectrals = cluster.SpectralClustering(n_clusters=num_clusters, assign_labels='discretize').fit(df.values)
    eval_cluster(labels, spectrals.labels_)

    print("GMM:")
    gmms = mixture.GaussianMixture(n_components=num_clusters).fit_predict(df.values)
    eval_cluster(labels, gmms)


def get_data_features(args: argparse.Namespace, modalities: List[str]):
    if args.load_features:
        return pd.read_csv(get_feature_file_name(args.experiment, args.excluded_modalities))
    else:
        curr_experiment = experiments_info[args.experiment]
        root_dir = args.experiment if args.experiment_path is None else args.experiment_path
        return extract_features(modalities, args.split_cycle, curr_experiment.start_date, curr_experiment.end_date,
                                args.experiment, root_dir, args.excluded_modalities)


if __name__ == '__main__':
    mods = list(experiments_info['EXP0'].modalities_norms.keys())
    parser = argparse.ArgumentParser(description='Run the clustering program.')
    subparsers = parser.add_subparsers(title='Subcommands', description='compare_clusters, plot_TSNE')

    # The subparser for the clustering
    clusters_parser = subparsers.add_parser('compare_clusters',
                                            help='Compare the cluster evaluation results for the chosen modalities.')
    clusters_parser.add_argument('-c', '--num_clusters', dest='num_clusters', type=int, default=6,
                                 help='The number of clusters used in the evaluations,')
    clusters_parser.add_argument('--exclude_modalities', '--exclude', dest='excluded_modalities', nargs='*',
                                 choices=mods, default=[],
                                 help=f"All of the modalities that you don't want to use. Choices are: {mods}")
    clusters_parser.add_argument('-l', '--load_features', dest='load_features', action='store_true', default=False,
                                 help="""Loads the features from a file when used.
                                 Otherwise they are computed from the extractor and saved in a csv file.""")
    add_experiment_dataset_arguments(clusters_parser)
    clusters_parser.set_defaults(
        func=lambda args: cluster_comp(
            get_data_features(args, mods),
            args.num_clusters
        )
    )

    # The subparser for tsne
    tsne_parser = subparsers.add_parser('plot_TSNE', help='Save a TSNE plot for the chosen modalities.')
    tsne_parser.add_argument('--exclude_modalities', '--exclude', dest='excluded_modalities', nargs='*', choices=mods,
                             default=[], help=f"All of the modalities that you don't want to use. Choices are: {mods}")
    tsne_parser.add_argument('-l', '--load_features', dest='load_features', action='store_true', default=False,
                             help="""Loads the features from a file when used.
                                    Otherwise they are computed from the extractor and saved in a csv file.""")
    tsne_parser.add_argument('--PCA', dest='PCA', type=int, default=0,
                             help="""If a positive number is inputted,
                             the features will be transformed by PCA with that number of components.
                             Otherwise there won't be any use of PCA. 0 by default.""")
    add_experiment_dataset_arguments(tsne_parser)
    tsne_parser.set_defaults(
        func=lambda args: plot_tsne(
            get_data_features(args, mods),
            args.experiment,
            args.excluded_modalities,
            args.PCA
        )
    )

    arguments = parser.parse_args()

    if len(vars(arguments)) == 0:
        print("Please call this program with the -h flag.")
    else:
        arguments.func(arguments)
