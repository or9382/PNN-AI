
from datetime import datetime
import os
import torch
from torch import nn, optim
from torch.utils import data
import torchvision.transforms as T

from datasets import Modalities, ModalitiesSubset, classes
from datasets.transformations import *
from model import PlantFeatureExtractor as FeatureExtractor


use_checkpoint = False

# training hyper-parameters
epochs = 25
label_lr = 1e-3
plant_lr = 1e-3
extractor_lr = 1e-3

domain_adapt_lr = 0.01

# dataset parameters
start_date = datetime(2019, 6, 5)
end_date = datetime(2019, 6, 19)
split_cycle = 7
lwir_max_len = 44
skip = 5
vir_max_len = 6

train_ratio = 3 / 4
batch_size = 4

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
    # 'polar': {'max_len': vir_max_len, 'transform': trans_polar}
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Modalities(
    'Exp0', split_cycle=split_cycle, start_date=start_date, end_date=end_date, **modalities
)

train_amount = int(train_ratio * dataset.num_plants)
test_amount = dataset.num_plants - train_amount

train_set, test_set = ModalitiesSubset.random_split(dataset, [train_amount, test_amount])
train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)

feat_ext = FeatureExtractor(*modalities).to(device)
label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes)).to(device))
plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, train_set.num_plants).to(device))

criterion = nn.CrossEntropyLoss().cuda()

label_opt = optim.Adam(label_cls.parameters(), lr=label_lr)
plant_opt = optim.Adam(plant_cls.parameters(), lr=plant_lr)
ext_opt = optim.Adam(feat_ext.parameters(), lr=extractor_lr)

best_loss = float('inf')


def test_model(save_checkpoints=True):
    test_loader = data.DataLoader(test_set, batch_size=batch_size, num_workers=2, shuffle=True)

    global best_loss
    print('\ttesting model:')

    feat_ext.eval()
    label_cls.eval()
    plant_cls.eval()

    tot_accuracy = 0.
    tot_label_loss = 0.
    with torch.no_grad():
        for batch in test_loader:

            for key in batch:
                batch[key] = batch[key].to(device)

            labels = batch['label']

            x = batch.copy()

            del x['label']
            del x['plant']

            features: torch.Tensor = feat_ext(**x)
            label_out = label_cls(features)
            label_loss = criterion(label_out, labels)

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_accuracy += equality.float().mean().item()
            tot_label_loss += label_loss.item()

    accuracy = tot_accuracy / (len(test_set) / batch_size)
    loss = tot_label_loss / (len(test_set) / batch_size)
    print(f"\t\tlabel accuracy - {accuracy}")
    print(f"\t\tlabel loss - {loss}")

    if save_checkpoints and loss < best_loss:
        best_loss = loss

        print(f'\t\tsaving model with new best loss {best_loss}')
        torch.save({
            'feat_ext_state_dict': feat_ext.state_dict(),
            'label_cls_state_dict': label_cls.state_dict(),
            'plant_cls_state_dict': plant_cls.state_dict(),
            'loss': best_loss,
            'accuracy': accuracy
        }, 'checkpoint')

    return tot_accuracy / len(test_set), tot_label_loss / len(test_set)


def train_loop(save_checkpoints=True):

    for epoch in range(epochs):
        print(f"epoch {epoch + 1}:")

        feat_ext.train()
        label_cls.train()
        plant_cls.train()

        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.
        for i, batch in enumerate(train_loader):

            for key in batch:
                batch[key] = batch[key].to(device)

            labels = batch['label']
            plants = batch['plant']

            x = batch.copy()

            del x['label']
            del x['plant']

            label_opt.zero_grad()
            plant_opt.zero_grad()
            ext_opt.zero_grad()

            features: torch.Tensor = feat_ext(**x)
            features_plants = features.clone()
            features_plants.register_hook(lambda grad: -domain_adapt_lr * grad)

            label_out = label_cls(features)
            label_loss = criterion(label_out, labels)

            plant_out = plant_cls(features_plants)
            plant_loss = criterion(plant_out, plants)
            (label_loss + plant_loss).backward()

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_accuracy += equality.float().mean()

            label_opt.step()
            ext_opt.step()
            plant_opt.step()

            tot_label_loss += label_loss.item()
            tot_plant_loss += plant_loss.item()

            if i % 24 == 23:
                print(f"\t{i}. label loss: {tot_label_loss / 24}")
                print(f"\t{i}. plant loss: {tot_plant_loss / 24}")
                print(f"\t{i}. accuracy: {tot_accuracy / 24}")

                tot_label_loss = 0.
                tot_plant_loss = 0.
                tot_accuracy = 0.

        test_model(save_checkpoints)

#
# def leave_one_out_test(amount=10):
#     global best_loss, feat_ext, label_cls, plant_cls, criterion, label_opt, plant_opt, ext_opt
#
#     with open("leave_one_out_results.txt", 'w') as f:
#
#         tot_accuracy = 0.
#         tot_loss = 0.
#         for i in list(range(dataset.num_plants))[:amount]:
#             print(f"plant left out: {i}")
#             out, rest = ModalitiesSubset.leave_one_out(dataset, i)
#
#             best_loss = float('inf')
#             feat_ext = FeatureExtractor(*modalities).to(device)
#             label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes)).to(device))
#             plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, 48).to(device))
#
#             criterion = nn.CrossEntropyLoss().cuda()
#
#             label_opt = optim.Adam(label_cls.parameters(), lr=label_lr)
#             plant_opt = optim.Adam(plant_cls.parameters(), lr=plant_lr)
#             ext_opt = optim.Adam(feat_ext.parameters(), lr=extractor_lr)
#
#             train_loop(rest)
#             restore_checkpoint()
#
#             print("testing left-out plant")
#             accuracy, loss = test_model(out, save_checkpoints=False)
#             tot_accuracy += accuracy
#             tot_loss += loss
#
#             f.write(f"{i}) accuracy: {accuracy}\n")
#             f.write(f"{i}) loss: {loss}\n")
#             f.flush()
#             os.fsync(f.fileno())
#             print()
#
#     print(f"Leave-One-Out results:")
#     print(f"accuracy - {tot_accuracy / dataset.num_plants}")
#     print(f"loss - {tot_loss / dataset.num_plants}")


def restore_checkpoint(checkpoint_name=None):
    if not checkpoint_name:
        checkpoint_name = 'checkpoint'

    global best_loss, feat_ext, label_cls, plant_cls

    checkpoint = torch.load(checkpoint_name)
    feat_ext.load_state_dict(checkpoint['feat_ext_state_dict'])
    label_cls.load_state_dict(checkpoint['label_cls_state_dict'])
    plant_cls.load_state_dict(checkpoint['plant_cls_state_dict'])
    best_loss = checkpoint['loss']

    print(f"restoring model to one with loss - {best_loss}")

    feat_ext = feat_ext.to(device)
    label_cls = label_cls.to(device)
    plant_cls = plant_cls.to(device)


if __name__ == '__main__':
    if use_checkpoint:
        restore_checkpoint()

    train_loop()
