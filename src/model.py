# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class AutoEncoder(nn.Module):

    def __init__(self, y_dim): #这里的y是[0,1]或是[1,0]标签的维数 2倍原来

        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Sequential( #[BS,256,256,3]->[BS,128,128,32]
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential( #[BS,128,128,32]->[BS,64,64,64]
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential( #[BS,64,64,64]->[BS,32,32,128]
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential( #[BS,32,32,128]->[BS,16,16,256]
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential( #[BS,16,16,256]->[BS,8,8,512]
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.conv6 = nn.Sequential( #[BS,8,8,512]->[BS,4,4,512]
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.deconv1 = nn.Sequential( #[BS,4,4,512+y_num]->[BS,8,8,512]
            nn.ConvTranspose2d(512 + y_dim, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential( #[BS,8,8,512+y_num]->[BS,16,16,256]
            nn.ConvTranspose2d(512 + y_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential( #[BS,16,16,256+y_num]->[BS,32,32,128]
            nn.ConvTranspose2d(256 + y_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential( #[BS,32,32,128+y_num]->[BS,64,64,64]
            nn.ConvTranspose2d(128 + y_dim, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv5 = nn.Sequential( #[BS,64,64,64+y_num]->[BS,128,128,32]
            nn.ConvTranspose2d(64 + y_dim, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv6 = nn.Sequential( #[BS,128,128,32+y_num]->[BS,256,256,3]
            nn.ConvTranspose2d(32 + y_dim, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    def decode(self, z, y):
        bs = z.size(0)
        y = y.unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, y.expand(bs, self.y_dim, 4, 4)], 1)
        x = self.deconv1(x)
        x = torch.cat([x, y.expand(bs, self.y_dim, 8, 8)], 1)
        x = self.deconv2(x)
        x = torch.cat([x, y.expand(bs, self.y_dim, 16, 16)], 1)
        x = self.deconv3(x)
        x = torch.cat([x, y.expand(bs, self.y_dim, 32, 32)], 1)
        x = self.deconv4(x)
        x = torch.cat([x, y.expand(bs, self.y_dim, 64, 64)], 1)
        x = self.deconv5(x)
        x = torch.cat([x, y.expand(bs, self.y_dim, 128, 128)], 1)
        x = self.deconv6(x)
        return x

    def forward(self, X, y):
        enc_output = self.encode(X)
        dec_output = self.decode(enc_output, y)
        return enc_output, dec_output


class LatentDiscriminator(nn.Module):

    def __init__(self, y_dim):
        super(LatentDiscriminator, self).__init__()
        self.conv1 = nn.Sequential( #[BS,512,4,4]–>[BS,512,2,2]
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )
        self.conv2 = nn.Sequential( #[BS,512,2,2]–>[BS,512,1,1]
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )
        self.fc1 = nn.Sequential( #[BS,512] ->[BS,512]
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential( #[BS,512] ->[BS,y_dim]
            nn.Linear(512, y_dim),
        )

    def forward(self, z):
        x = self.conv1(z)
        x = self.conv2(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Sequential( #[BS,3,256,256]->[BS,32,128,128]
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential( #[BS,32,128,128]->[BS,64,64,64]
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential( #[BS,64,64,64]->[BS,128,32,32]
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential( #[BS,128,32,32]->BS,256,32,32]
            nn.Conv2d(128, 256, 4, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential( #[BS,256,32,32]->[BS,1,32,32]
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1).mean(1).view(x.size(0))
        return x


class Classifier(nn.Module):

    def __init__(self, y_dim):
        super(Classifier, self).__init__()
        self.conv1 = nn.Sequential( #[BS,3,256,256]->[BS,32,128,128]
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential( #[BS,32,128,128]->[BS,64,64,64]
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential( #[BS,64,64,64]->[BS,128,32,32]
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential( #[BS,128,32,32]->[BS,256,16,16]
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential( #[BS,256,16,16]->[BS,512,8,8]
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.conv6 = nn.Sequential( #[BS,512,8,8]->[BS,512,4,4]
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.conv7 = nn.Sequential( #[BS,512,4,4]->[BS,512,2,2]
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.conv8 = nn.Sequential( #[BS,512,2,2]->[BS,512,1,1]
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.fc1 = nn.Sequential( #[BS,512]->[BS,512]
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential( #[BS,512]->[BS,y_dim]
            nn.Linear(512, y_dim)
        )

    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_attr_loss(output, attributes, flip, params):
    """
    Compute attributes loss.
    """
    assert type(flip) is bool
    k = 0
    loss = 0
    for (_, n_cat) in params.attr:
        # categorical
        x = output[:, k:k + n_cat].contiguous()
        y = attributes[:, k:k + n_cat].max(1)[1].view(-1)
        if flip:
            # generate different categories
            shift = torch.LongTensor(y.size()).random_(n_cat - 1) + 1
            y = (y + Variable(shift.cuda())) % n_cat
        loss += F.cross_entropy(x, y)
        k += n_cat
    return loss


def update_predictions(all_preds, preds, targets, params):
    """
    Update discriminator / classifier predictions.
    """
    assert len(all_preds) == len(params.attr)
    k = 0
    for j, (_, n_cat) in enumerate(params.attr):
        _preds = preds[:, k:k + n_cat].max(1)[1]
        _targets = targets[:, k:k + n_cat].max(1)[1]
        all_preds[j].extend((_preds == _targets).tolist())
        k += n_cat
    assert k == params.n_attr


def get_mappings(params):
    """
    Create a mapping between attributes and their associated IDs.
    """
    if not hasattr(params, 'mappings'):
        mappings = []
        k = 0
        for (_, n_cat) in params.attr:
            assert n_cat >= 2
            mappings.append((k, k + n_cat))
            k += n_cat
        assert k == params.n_attr
        params.mappings = mappings
    return params.mappings


def flip_attributes(attributes, params, attribute_id, new_value=None):
    """
    Randomly flip a set of attributes.
    """
    assert attributes.size(1) == params.n_attr
    mappings = get_mappings(params)
    attributes = attributes.data.clone().cpu()

    def flip_attribute(attribute_id, new_value=None):
        bs = attributes.size(0)
        i, j = mappings[attribute_id]
        attributes[:, i:j].zero_()
        if new_value is None:
            y = torch.LongTensor(bs).random_(j - i)
        else:
            assert new_value in range(j - i)
            y = torch.LongTensor(bs).fill_(new_value)
        attributes[:, i:j].scatter_(1, y.unsqueeze(1), 1)

    if attribute_id == 'all':
        assert new_value is None
        for attribute_id in range(len(params.attr)):
            flip_attribute(attribute_id)
    else:
        assert type(new_value) is int
        flip_attribute(attribute_id, new_value)

    return Variable(attributes.cuda())
