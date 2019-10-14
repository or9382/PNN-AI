from datetime import datetime
import os
import torch
from torch import nn, optim
from torch.utils import data
import torchvision.transforms as T
import argparse

from datasets import Modalities, ModalitiesSubset, classes
from datasets.transformations import *
from model import PlantFeatureExtractor as FeatureExtractor
from .utils import get_checkpoint_name, get_used_modalities


# define test config
class TestConfig:
    def __init__(self, use_checkpoint, checkpoint_name, epochs, batch_size, domain_adapt_lr, device, dataset, train_set,
                 test_set,
                 train_loader, feat_ext, label_cls, plant_cls, criterion, label_opt, plant_opt, ext_opt, best_loss):
        self.use_checkpoint = use_checkpoint
        self.checkpoint_name = checkpoint_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.domain_adapt_lr = domain_adapt_lr
        self.device = device
        self.dataset = dataset
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = train_loader
        self.epochs = epochs
        self.feat_ext = feat_ext
        self.label_cls = label_cls
        self.plant_cls = plant_cls
        self.criterion = criterion
        self.label_opt = label_opt
        self.plant_opt = plant_opt
        self.ext_opt = ext_opt
        self.best_loss = best_loss


# dataset parameters
start_date = datetime(2019, 6, 5)
end_date = datetime(2019, 6, 19)
split_cycle = 7
lwir_max_len = 44
skip = 5
vir_max_len = 6

trans_lwir = T.Compose([
    T.Normalize([21361.], [481.]), T.ToPILImage(),
    RandomCrop(lwir_max_len, (206, 206)), RandomHorizontalFlip(lwir_max_len),
    RandomVerticalFlip(lwir_max_len), T.ToTensor()
])

trans_577 = T.Compose([
    T.Normalize([.00607], [.00773]), T.ToPILImage(),
    RandomCrop(vir_max_len, (412, 412)), RandomHorizontalFlip(vir_max_len),
    RandomVerticalFlip(vir_max_len), T.ToTensor()
])

trans_692 = T.Compose([
    T.Normalize([.02629], [.04364]), T.ToPILImage(),
    RandomCrop(vir_max_len, (412, 412)), RandomHorizontalFlip(vir_max_len),
    RandomVerticalFlip(vir_max_len), T.ToTensor()
])

trans_732 = T.Compose([
    T.Normalize([.01072], [.11680]), T.ToPILImage(),
    RandomCrop(vir_max_len, (412, 412)), RandomHorizontalFlip(vir_max_len),
    RandomVerticalFlip(vir_max_len), T.ToTensor()
])

trans_970 = T.Compose([
    T.Normalize([.00125], [.00095]), T.ToPILImage(),
    RandomCrop(vir_max_len, (412, 412)), RandomHorizontalFlip(vir_max_len),
    RandomVerticalFlip(vir_max_len), T.ToTensor()
])

trans_polar = T.Compose([
    T.Normalize([.05136], [.22331]), T.ToPILImage(),
    RandomCrop(vir_max_len, (412, 412)), RandomHorizontalFlip(vir_max_len),
    RandomVerticalFlip(vir_max_len), T.ToTensor()
])

modalities = {
    'lwir': {'max_len': lwir_max_len, 'skip': skip, 'transform': trans_lwir},
    '577nm': {'max_len': vir_max_len, 'transform': trans_577},
    '692nm': {'max_len': vir_max_len, 'transform': trans_692},
    '732nm': {'max_len': vir_max_len, 'transform': trans_732},
    '970nm': {'max_len': vir_max_len, 'transform': trans_970},
    'polar': {'max_len': vir_max_len, 'transform': trans_polar}
}


def test_model(test_config: TestConfig):
    test_loader = data.DataLoader(test_config.test_set, batch_size=test_config.batch_size, num_workers=2, shuffle=True)

    print('\ttesting model:')

    test_config.feat_ext.eval()
    test_config.label_cls.eval()
    test_config.plant_cls.eval()

    tot_accuracy = 0.
    tot_label_loss = 0.
    with torch.no_grad():
        for batch in test_loader:

            for key in batch:
                batch[key] = batch[key].to(test_config.device)

            labels = batch['label']

            x = batch.copy()

            del x['label']
            del x['plant']

            features: torch.Tensor = test_config.feat_ext(**x)
            label_out = test_config.label_cls(features)
            label_loss = test_config.criterion(label_out, labels)

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_accuracy += equality.float().mean().item()
            tot_label_loss += label_loss.item()

    accuracy = tot_accuracy / (len(test_config.test_set) / test_config.batch_size)
    loss = tot_label_loss / (len(test_config.test_set) / test_config.batch_size)
    print(f"\t\tlabel accuracy - {accuracy}")
    print(f"\t\tlabel loss - {loss}")

    if test_config.use_checkpoint and loss < test_config.best_loss:
        test_config.best_loss = loss

        print(f'\t\tsaving model with new best loss {test_config.best_loss}')
        torch.save({
            'feat_ext_state_dict': test_config.feat_ext.state_dict(),
            'label_cls_state_dict': test_config.label_cls.state_dict(),
            'plant_cls_state_dict': test_config.plant_cls.state_dict(),
            'loss': test_config.best_loss,
            'accuracy': accuracy
        }, 'checkpoint')

    return tot_accuracy / len(test_config.test_set), tot_label_loss / len(test_config.test_set)


def train_loop(test_config: TestConfig):
    for epoch in range(test_config.epochs):
        print(f"epoch {epoch + 1}:")

        test_config.feat_ext.train()
        test_config.label_cls.train()
        test_config.plant_cls.train()

        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.
        for i, batch in enumerate(test_config.train_loader):

            for key in batch:
                batch[key] = batch[key].to(test_config.device)

            labels = batch['label']
            plants = batch['plant']

            x = batch.copy()

            del x['label']
            del x['plant']

            test_config.label_opt.zero_grad()
            test_config.plant_opt.zero_grad()
            test_config.ext_opt.zero_grad()

            features: torch.Tensor = test_config.feat_ext(**x)
            features_plants = features.clone()
            features_plants.register_hook(lambda grad: -test_config.domain_adapt_lr * grad)

            label_out = test_config.label_cls(features)
            label_loss = test_config.criterion(label_out, labels)

            plant_out = test_config.plant_cls(features_plants)
            plant_loss = test_config.criterion(plant_out, plants)
            (label_loss + plant_loss).backward()

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_accuracy += equality.float().mean()

            test_config.label_opt.step()
            test_config.ext_opt.step()
            test_config.plant_opt.step()

            tot_label_loss += label_loss.item()
            tot_plant_loss += plant_loss.item()

            if i % 24 == 23:
                print(f"\t{i}. label loss: {tot_label_loss / 24}")
                print(f"\t{i}. plant loss: {tot_plant_loss / 24}")
                print(f"\t{i}. accuracy: {tot_accuracy / 24}")

                tot_label_loss = 0.
                tot_plant_loss = 0.
                tot_accuracy = 0.

        test_model(test_config)


def restore_checkpoint(test_config: TestConfig):
    checkpoint = torch.load(test_config.checkpoint_name)
    test_config.feat_ext.load_state_dict(checkpoint['feat_ext_state_dict'])
    test_config.label_cls.load_state_dict(checkpoint['label_cls_state_dict'])
    test_config.plant_cls.load_state_dict(checkpoint['plant_cls_state_dict'])
    test_config.best_loss = checkpoint['loss']

    print(f"Restoring model to one with loss - {test_config.best_loss}")

    test_config.feat_ext = test_config.feat_ext.to(test_config.device)
    test_config.label_cls = test_config.label_cls.to(test_config.device)
    test_config.plant_cls = test_config.plant_cls.to(test_config.device)


def main(args: argparse.Namespace):
    checkpoint_name = get_checkpoint_name(args.excluded_modalities)

    # training hyper-parameters
    epochs = args.epochs
    batch_size = args.batch_size
    label_lr = args.label_lr
    plant_lr = args.plant_lr
    extractor_lr = args.extractor_lr

    domain_adapt_lr = args.domain_adapt_lr

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    used_modalities = get_used_modalities(modalities, args.excluded_modalities)

    dataset = Modalities('Exp0', split_cycle=split_cycle, start_date=start_date, end_date=end_date, **used_modalities)

    train_amount = int(args.train_ratio * dataset.num_plants)
    test_amount = dataset.num_plants - train_amount

    train_set, test_set = ModalitiesSubset.random_split(dataset, [train_amount, test_amount])
    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)

    feat_ext = FeatureExtractor(*used_modalities).to(device)
    label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes)).to(device))
    plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, train_set.num_plants).to(device))

    criterion = nn.CrossEntropyLoss().cuda()

    label_opt = optim.Adam(label_cls.parameters(), lr=label_lr)
    plant_opt = optim.Adam(plant_cls.parameters(), lr=plant_lr)
    ext_opt = optim.Adam(feat_ext.parameters(), lr=extractor_lr)

    best_loss = float('inf')

    test_config = TestConfig(args.use_checkpoint, checkpoint_name, epochs, batch_size, domain_adapt_lr, device, dataset,
                             train_set, test_set, train_loader, feat_ext, label_cls, plant_cls, criterion, label_opt,
                             plant_opt, ext_opt, best_loss)

    if args.use_checkpoint:
        restore_checkpoint(test_config)

    train_loop(test_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the train loop.')
    parser.add_argument('-c', '--use_checkpoint', dest='use_checkpoint', action='store_true', default=False,
                        help='Flag for using checkpoints in the training')
    mods = list(modalities.keys())
    parser.add_argument('--excluded_modalities', dest='excluded_modalities', nargs='?', choices=mods, default=[],
                        help=f"All of the modalities that you don't want to use. Choices are: {mods}")
    parser.add_argument('-e', '--epochs', dest='epochs', default=25, type=int,
                        help='The number of epochs used in the training.')
    parser.add_argument('--domain_adapt_lr', dest='domain_adapt_lr', type=float, default=0.01,
                        help='The coefficient used in the domain adaptation.')
    parser.add_argument('--label_lr', dest='label_lr', type=float, default=1e-3,
                        help='The learning rate for the phenotype classifier.')
    parser.add_argument('--plant_lr', dest='plant_lr', type=float, default=1e-3,
                        help='The learning rate for the plant classifier used in the transfer learning.')
    parser.add_argument('--extractor_lr', dest='extractor_lr', type=float, default=1e-3,
                        help='The learning rate for the feature extractor.')
    parser.add_argument('-t', '--train_ratio', dest='train_ratio', type=float, default=0.75,
                        help='The ratio of the dataset that will be used for training.')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=4,
                        help='The batch size for the training.')

    arguments = parser.parse_args()
    main(arguments)
