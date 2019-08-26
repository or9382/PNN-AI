
import torch
from torch import nn, optim
from torch.utils import data

from datasets import Modalities, classes
from model import LinearWrapper, PlantFeatureExtractor as FeatureExtractor

epochs = 5
label_lr = 0.001
plant_lr = 0.001
modalities = {
    '577nm': {'max_len': 36},
    'polar': {'max_len': 36}
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Modalities('Exp0', split_cycle=7, **modalities)
dataloader = data.DataLoader(dataset, batch_size=4, num_workers=2)

feat_ext = FeatureExtractor(*modalities)
label_cls = LinearWrapper(feat_ext, 512, len(classes))
plant_cls = LinearWrapper(feat_ext, 512, 48)

criterion = nn.CrossEntropyLoss()
label_opt = optim.SGD(label_cls.parameters(), lr=label_lr)
plant_opt = optim.SGD(plant_cls.parameters(), lr=plant_lr)

for epoch in range(epochs):
    print(f"epoch {epoch + 1} - ", end='')

    running_loss = 0.
    for i, batch in enumerate(dataloader):
        print(i)

        labels = batch['label']
        plants = batch['plant']

        del batch['label']
        del batch['plant']

        label_opt.zero_grad()
        plant_opt.zero_grad()

        label_out = label_cls(**batch)
        label_loss = criterion(label_out, labels)
        label_loss.backward()
        label_opt.step()

        plant_out = plant_cls(**batch)
        plant_loss = criterion(plant_out, plants)
        (-plant_loss).backward()
        plant_opt.step()

    print(f"loss: {running_loss}")
