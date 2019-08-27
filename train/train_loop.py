
import torch
from torch import nn, optim
from torch.utils import data

from datasets import Modalities, classes
from model import LinearWrapper, PlantFeatureExtractor as FeatureExtractor

epochs = 5
label_lr = 0.001
plant_lr = 0.001
extractor_lr = 0.001

domain_adapt_l = 0.01

modalities = {
    '577nm': {'max_len': 36},
    'polar': {'max_len': 36}
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Modalities('Exp0', split_cycle=7, **modalities)
dataloader = data.DataLoader(dataset, batch_size=4, num_workers=2)

feat_ext = FeatureExtractor(*modalities)
label_cls = nn.Sequential(nn.Linear(512, len(classes)), nn.Softmax())
plant_cls = nn.Sequential(nn.Linear(512, 48), nn.Softmax())

plant_cls.register_backward_hook(lambda grad: -domain_adapt_l * grad)

criterion = nn.CrossEntropyLoss()

label_opt = optim.adam.Adam(label_cls.parameters(), lr=label_lr)
plant_opt = optim.adam.Adam(plant_cls.parameters(), lr=plant_lr)
ext_opt = optim.adam.Adam(plant_cls.parameters(), lr=extractor_lr)

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
        ext_opt.zero_grad()

        features = feat_ext(**batch)

        label_out = label_cls(features)
        label_loss = criterion(label_out, labels)
        label_loss.backward()

        plant_out = plant_cls(features)
        plant_loss = criterion(plant_out, plants)
        plant_loss.backward()

        label_opt.step()
        ext_opt.step()
        plant_opt.step()


    print(f"loss: {running_loss}")
