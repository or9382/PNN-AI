
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils import data
import torchvision.transforms as T

from datasets import Modalities, classes
from datasets.transformations import *
from model import PlantFeatureExtractor as FeatureExtractor

save_checkpoints = True
use_checkpoint = False

# training hyper-parameters
epochs = 10
label_lr = 1e-4
plant_lr = 1e-4
extractor_lr = 1e-4

domain_adapt_lr = 1e-2

# dataset parameters
start_date = datetime(2019, 6, 5)
end_date = datetime(2019, 6, 23)
split_cycle = 7
lwir_max_len = 250
vir_max_len = 8

train_ration = 5 / 6

trans_lwir = T.Compose([
    T.Normalize([21361.], [481.]), T.ToPILImage(),
    RandomCrop(lwir_max_len, (229, 229)), T.ToTensor()
])

trans_577 = T.Compose([
    T.Normalize([12827.], [9353.]), T.ToPILImage(),
    RandomCrop(vir_max_len, (458, 458)), T.ToTensor()
])

trans_692 = T.Compose([
    T.Normalize([12650.], [12021.]), T.ToPILImage(),
    RandomCrop(vir_max_len, (458, 458)), T.ToTensor()
])

trans_732 = T.Compose([
    T.Normalize([3169.], [21595.]), T.ToPILImage(),
    RandomCrop(vir_max_len, (458, 458)), T.ToTensor()
])

trans_970 = T.Compose([
    T.Normalize([7389.], [4291.]), T.ToPILImage(),
    RandomCrop(vir_max_len, (458, 458)), T.ToTensor()
])

trans_polar = T.Compose([
    T.Normalize([6248.], [22033.]), T.ToPILImage(),
    RandomCrop(vir_max_len, (458, 458)), T.ToTensor()
])

modalities = {
    'lwir': {'img_len': 300, 'max_len': lwir_max_len, 'transform': trans_lwir},
    '577nm': {'img_len': 500, 'max_len': vir_max_len, 'transform': trans_577},
    '692nm': {'img_len': 500, 'max_len': vir_max_len, 'transform': trans_692},
    '732nm': {'img_len': 500, 'max_len': vir_max_len, 'transform': trans_732},
    '970nm': {'img_len': 500, 'max_len': vir_max_len, 'transform': trans_970},
    'polar': {'img_len': 500, 'max_len': vir_max_len, 'transform': trans_polar}
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Modalities(
    'Exp0', split_cycle=split_cycle, start_date=start_date, end_date=end_date, **modalities
)

train_amount = int(train_ration * len(dataset))
test_amount = len(dataset) - train_amount

train_set, test_set = data.random_split(dataset, (train_amount, test_amount))

train_loader = data.DataLoader(train_set, batch_size=4, num_workers=2, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=4, num_workers=2, shuffle=True)

feat_ext = FeatureExtractor(*modalities).to(device)
label_cls = nn.Linear(512, len(classes)).to(device)
plant_cls = nn.Linear(512, 48).to(device)

criterion = nn.CrossEntropyLoss()

label_opt = optim.Adam(label_cls.parameters(), lr=label_lr)
plant_opt = optim.Adam(plant_cls.parameters(), lr=plant_lr)
ext_opt = optim.Adam(feat_ext.parameters(), lr=extractor_lr)

best_loss = float('inf')


def test_model():
    global best_loss
    print('testing model:')

    feat_ext.eval()
    label_cls.eval()
    plant_cls.eval()

    tot_label_correct = 0
    tot_plant_correct = 0
    tot_label_loss = 0.
    tot_plant_loss = 0.
    with torch.no_grad():
        for batch in test_loader:

            for key in batch:
                batch[key] = batch[key].to(device)

            labels = batch['label']
            plants = batch['plant']

            x = batch.copy()

            del x['label']
            del x['plant']

            features: torch.Tensor = feat_ext(**x)
            label_out = label_cls(features)
            plant_out = plant_cls(features)

            label_equality = (labels.data == label_out.max(dim=1)[1])
            plant_equality = (plants.data == plant_out.max(dim=1)[1])

            tot_label_correct += sum(label_equality).item()
            tot_plant_correct += sum(plant_equality).item()

            tot_label_loss += criterion(label_out, labels).item()
            tot_plant_loss += criterion(plant_out, plants).item()

    print(f"\tlabel accuracy - {tot_label_correct / (len(test_set))}")
    print(f"\tlabel loss - {4 * tot_label_loss / (len(test_set))}")
    print(f"\tplant accuracy - {tot_plant_correct / (len(test_set))}")
    print(f"\tplant loss - {4 * tot_plant_loss / (len(test_set))}")

    if save_checkpoints and tot_label_loss / tot_plant_loss < best_loss:
        best_loss = tot_label_loss / tot_plant_loss

        torch.save({
            'feat_ext_state_dict': feat_ext.state_dict(),
            'label_cls_state_dict': label_cls.state_dict(),
            'plant_cls_state_dict': plant_cls.state_dict(),
            'loss': best_loss
        }, 'checkpoint')


def train_loop():
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

            if i % 6 == 5:
                print(f"\t{i}. label loss: {tot_label_loss / 6}")
                print(f"\t{i}. plant loss: {tot_plant_loss / 6}")
                print(f"\t{i}. accuracy: {tot_accuracy / 6}")

                tot_label_loss = 0.
                tot_plant_loss = 0.
                tot_accuracy = 0.

        test_model()


def restore_checkpoint():
    global best_loss, feat_ext, label_cls, plant_cls

    checkpoint = torch.load('checkpoint')
    feat_ext.load_state_dict(checkpoint['feat_ext_state_dict'])
    label_cls.load_state_dict(checkpoint['label_cls_state_dict'])
    plant_cls.load_state_dict(checkpoint['plant_cls_state_dict'])
    best_loss = checkpoint['loss']

    feat_ext = feat_ext.to(device)
    label_cls = label_cls.to(device)
    plant_cls = plant_cls.to(device)


if __name__ == '__main__':
    if use_checkpoint:
        restore_checkpoint()

    train_loop()
