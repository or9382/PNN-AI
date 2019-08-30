
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils import data
from torchvision.transforms import Normalize

from datasets import Modalities, classes
from model import LinearWrapper, PlantFeatureExtractor as FeatureExtractor

# training hyper-parameters
epochs = 10
label_lr = 1e-3
plant_lr = 1e-3
extractor_lr = 1e-3

domain_adapt_lr = 0.01

# dataset parameters
start_date = datetime(2019, 6, 5)
end_date = datetime(2019, 6, 23)
split_cycle = 7
lwir_max_len = 250
vir_max_len = 8

train_ration = 5 / 6

modalities = {
    'lwir': {'max_len': lwir_max_len, 'transform': Normalize([21361.], [481.])},
    '577nm': {'max_len': vir_max_len, 'transform': Normalize([12827.], [9353.])},
    '692nm': {'max_len': vir_max_len, 'transform': Normalize([12650.], [12021.])},
    '732nm': {'max_len': vir_max_len, 'transform': Normalize([3169.], [21595.])},
    '970nm': {'max_len': vir_max_len, 'transform': Normalize([7389.], [4291.])},
    'polar': {'max_len': vir_max_len, 'transform': Normalize([6248.], [22033.])}
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


def test_model():
    print('testing model:')

    feat_ext.eval()
    label_cls.eval()

    tot_correct = 0
    tot_loss = 0.
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

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_correct += sum(equality).item()
            tot_loss += criterion(label_out, labels).item()

    print(f"\taccuracy - {tot_correct / (len(test_set))}")
    print(f"\tloss - {4 * tot_loss / (len(test_set))}")


def train_loop():
    best_loss = float('inf')
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
            # label_loss.backward()

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

                if tot_label_loss < best_loss:
                    torch.save({
                        'epoch': epoch,
                        'feat_ext_state_dict': feat_ext.state_dict(),
                        'label_cls_state_dict': label_cls.state_dict(),
                        'plant_cls_state_dict': plant_cls.state_dict(),
                        'loss': tot_label_loss
                    }, f'checkpoint')

                    best_loss = tot_label_loss

                tot_label_loss = 0.
                tot_plant_loss = 0.
                tot_accuracy = 0.

        test_model()

if __name__ == '__main__':
    train_loop()