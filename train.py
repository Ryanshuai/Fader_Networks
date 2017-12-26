# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

from src.utils import initialize_log
from src.model import AutoEncoder, LatentDiscriminator, PatchDiscriminator, Classifier
from src.training import Trainer
from src.evaluation import Evaluator

from src.dataset import Dataset
from torch.utils.data import DataLoader

# parse parameters
parser = argparse.ArgumentParser(description='Images autoencoder')
parser.add_argument("--log_name", type=str, default="log",
                    help="Experiment name")
parser.add_argument("--attr", type=list, default=["Smiling","Male"],
                    help="Attributes to classify")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--n_epochs", type=int, default=1000,
                    help="Total number of epochs")
parser.add_argument("--epoch_size", type=int, default=32*25*5,
                    help="Number of samples per epoch")
parser.add_argument("--ae_reload", type=str, default="",
                    help="Reload a pretrained encoder")
parser.add_argument("--lat_dis_reload", type=str, default="",
                    help="Reload a pretrained latent discriminator")
parser.add_argument("--ptc_dis_reload", type=str, default="",
                    help="Reload a pretrained patch discriminator")
parser.add_argument("--clf_dis_reload", type=str, default="",
                    help="Reload a pretrained classifier discriminator")
params = parser.parse_args()
params.n_attr = len(params.attr)

logger = initialize_log(params)
# load Dataset
train_dataset = Dataset(params,'train')
val_dataset = Dataset(params, 'val')
train_data = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=1) # num_workers 几个线程参与读数据
valid_data = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=1)

# build the model
ae = AutoEncoder(params.n_attr).cuda()
lat_dis = LatentDiscriminator(params.n_attr).cuda()
ptc_dis = PatchDiscriminator().cuda()
clf_dis = Classifier(params.n_attr).cuda()

# trainer / evaluator
trainer = Trainer(ae, lat_dis, ptc_dis, clf_dis, train_data, params)
evaluator = Evaluator(ae, lat_dis, ptc_dis, clf_dis, valid_data, params)


for n_epoch in range(params.n_epochs):

    logger.info('Starting epoch %i...' % n_epoch)

    for n_iter in range(0, params.epoch_size, params.batch_size):

        # latent discriminator training
        trainer.lat_dis_step()

        # patch discriminator training
        trainer.ptc_dis_step()

        # classifier discriminator training
        trainer.clf_dis_step()

        # autoencoder training
        trainer.autoencoder_step()

        # print training statistics
        trainer.printLoss(n_iter)

    # run all evaluations / save best or periodic model
    to_log = evaluator.evaluate(n_epoch)
    trainer.save_best_periodic(to_log)
    logger.info('End of epoch %i.\n' % n_epoch)
