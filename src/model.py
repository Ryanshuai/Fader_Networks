# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

class AutoEncoder(nn.Module):

    def __init__(self, y_dim): #这里的y是[0,1]或是[1,0]标签的维数 2倍原
        super(AutoEncoder, self).__init__()
        self.y_dim = y_dim

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
            nn.ConvTranspose2d(512 + 2 * y_dim, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential( #[BS,8,8,512+y_num]->[BS,16,16,256]
            nn.ConvTranspose2d(512 + 2 * y_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential( #[BS,16,16,256+y_num]->[BS,32,32,128]
            nn.ConvTranspose2d(256 + 2 * y_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential( #[BS,32,32,128+y_num]->[BS,64,64,64]
            nn.ConvTranspose2d(128 + 2 * y_dim, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv5 = nn.Sequential( #[BS,64,64,64+y_num]->[BS,128,128,32]
            nn.ConvTranspose2d(64 + 2 * y_dim, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv6 = nn.Sequential( #[BS,128,128,32+y_num]->[BS,256,256,3]
            nn.ConvTranspose2d(32 + 2 * y_dim, 3, 4, stride=2, padding=1),
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
        BS = z.size(0)
        y = y.view(BS, -1)
        y = y.unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, y.expand(BS, 2 * self.y_dim, 4, 4)], 1)
        x = self.deconv1(x)
        x = torch.cat([x, y.expand(BS, 2 * self.y_dim, 8, 8)], 1)
        x = self.deconv2(x)
        x = torch.cat([x, y.expand(BS, 2 * self.y_dim, 16, 16)], 1)
        x = self.deconv3(x)
        x = torch.cat([x, y.expand(BS, 2 * self.y_dim, 32, 32)], 1)
        x = self.deconv4(x)
        x = torch.cat([x, y.expand(BS, 2 * self.y_dim, 64, 64)], 1)
        x = self.deconv5(x)
        x = torch.cat([x, y.expand(BS, 2 * self.y_dim, 128, 128)], 1)
        x = self.deconv6(x)
        return x

    def forward(self, X, y):
        enc_output = self.encode(X)
        dec_output = self.decode(enc_output, y)
        return enc_output, dec_output


class LatentDiscriminator(nn.Module):

    def __init__(self, y_dim):
        self.y_dim = y_dim
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
            nn.Linear(512, y_dim * 2),
        )

    def forward(self, z):
        x = self.conv1(z)
        x = self.conv2(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.y_dim , 2)
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
        self.y_dim = y_dim
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
            nn.Linear(512, y_dim * 2)
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
        x = self.fc2(x).view(-1, self.y_dim, 2)
        return x
