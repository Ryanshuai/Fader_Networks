# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from logging import getLogger
from torch.nn.utils import clip_grad_norm

from .utils import get_rand_attributes,softmax_cross_entropy
from  .utils import MODELS_PATH

logger = getLogger()


class Trainer(object):

    """ Trainer initialization. """
    def __init__(self, ae, lat_dis, ptc_dis, clf_dis, data, params):

        # data / parameters
        self.data = data
        self.data_iter = iter(data)


        # model params
        self.n_attr = params.n_attr
        self.clip_grad_norm = 5
        self.smooth_label = 0.2
        self.lambda_ae = 1
        self.lambda_lat_dis = 0.0001
        self.lambda_ptc_dis = 0.0001
        self.lambda_clf_dis = 0.0001
        self.lambda_final_step = 500000
        self.n_step = 0
        # models
        self.ae = ae
        self.lat_dis = lat_dis
        self.ptc_dis = ptc_dis
        self.clf_dis = clf_dis
        # logger model information
        logger.info(ae)
        logger.info('%i parameters in the autoencoder. ' % sum([p.nelement() for p in ae.parameters()]))
        logger.info(lat_dis)
        logger.info('%i parameters in the latent discriminator. ' % sum([p.nelement() for p in lat_dis.parameters()]))
        logger.info(ptc_dis)
        logger.info('%i parameters in the patch discriminator. ' % sum([p.nelement() for p in ptc_dis.parameters()]))
        logger.info(clf_dis)
        logger.info('%i parameters in the classifier discriminator. ' % sum([p.nelement() for p in clf_dis.parameters()]))

        # optimizers
        lr = 0.0002
        betas = (0.5, 0.999)
        self.ae_optimizer = torch.optim.Adam(ae.parameters(), lr=lr, betas=betas)
        self.lat_dis_optimizer = torch.optim.Adam(lat_dis.parameters(), lr=lr, betas=betas)
        self.ptc_dis_optimizer = torch.optim.Adam(ptc_dis.parameters(), lr=lr, betas=betas)
        self.clf_dis_optimizer = torch.optim.Adam(clf_dis.parameters(), lr=lr, betas=betas)

        # reload pretrained models
        if params.ae_reload:
            self.reload_model(ae, params.ae_reload)
        if params.lat_dis_reload:
            self.reload_model(lat_dis, params.lat_dis_reload)
        if params.ptc_dis_reload:
            self.reload_model(ptc_dis, params.ptc_dis_reload)
        if params.clf_dis_reload:
            self.reload_model(clf_dis, params.clf_dis_reload)

        # training statistics
        self.stats = {}
        self.stats['rec_costs'] = []
        self.stats['lat_dis_costs'] = []
        self.stats['ptc_dis_costs'] = []
        self.stats['clf_dis_costs'] = []

        # best reconstruction loss / best accuracy
        self.best_loss = 1e12
        self.best_accu = -1e12

    """ Train the latent discriminator. """
    def lat_dis_step(self):

        # read batch_data, 判断迭代器是否为空, 不空则取值
        try:
            batch_x, batch_y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data)
            batch_x, batch_y = next(self.data_iter)
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()


        self.ae.eval()
        self.lat_dis.train()

        # encode / discriminate
        enc_output = self.ae.encode(Variable(batch_x, volatile=True)) #只训练discriminator，所以volatile
        preds = self.lat_dis(Variable(enc_output.data))

        # loss / optimize
        loss = softmax_cross_entropy(preds, Variable(batch_y)) #训练lat_dis的预测结果接近batch_y
        self.stats['lat_dis_costs'].append(loss.data[0])
        self.lat_dis_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.lat_dis.parameters(), self.clip_grad_norm)  #!!!discriminator需要clip, torch.nn.utils.clip_grad_norm
        self.lat_dis_optimizer.step()

    """ Train the patch discriminator. autoencoder loss from the patch discriminator中产生对抗训练  """
    def ptc_dis_step(self):

        # read batch_data, 判断迭代器是否为空, 不空则取值
        try:
            batch_x, _ = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data)
            batch_x, _ = next(self.data_iter)
        batch_x = batch_x.cuda()

        self.ae.eval()
        self.ptc_dis.train()

        # encode / discriminate
        flipped = get_rand_attributes(batch_x.size(0), self.n_attr)
        _, dec_output = self.ae(Variable(batch_x, volatile=True), flipped)
        real_preds = self.ptc_dis(Variable(batch_x))
        fake_preds = self.ptc_dis(Variable(dec_output.data))
        y_fake = Variable(torch.FloatTensor(real_preds.size()).fill_(self.smooth_label).cuda())

        # loss / optimize
        loss = F.binary_cross_entropy(real_preds, 1 - y_fake) #让真的数据ptc_dis的结果更接近1-y_fake
        loss += F.binary_cross_entropy(fake_preds, y_fake) #让假的数据ptc_dis的结果更接近y_fake
        self.stats['ptc_dis_costs'].append(loss.data[0])
        self.ptc_dis_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.ptc_dis.parameters(), self.clip_grad_norm)
        self.ptc_dis_optimizer.step() #优化的参数是patch_discriminator的参数 #

    """ Train the classifier discriminator. #就是个多分类器的训练过程 """
    def clf_dis_step(self):

        # read batch_data, 判断迭代器是否为空, 不空则取值
        try:
            batch_x, batch_y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data)
            batch_x, batch_y = next(self.data_iter)
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        self.clf_dis.train()

        #  predict
        preds = self.clf_dis(Variable(batch_x))

        # loss / optimize
        loss = softmax_cross_entropy(preds, Variable(batch_y))
        self.stats['clf_dis_costs'].append(loss.data[0])
        self.clf_dis_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.clf_dis.parameters(), self.clip_grad_norm)
        self.clf_dis_optimizer.step()

    """ Train the autoencoder with cross-entropy loss. Train the encoder with discriminator loss. """
    def autoencoder_step(self):
        # read batch_data, 判断迭代器是否为空, 不空则取值
        try:
            batch_x, batch_y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data)
            batch_x, batch_y = next(self.data_iter)
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        self.ae.train()
        self.lat_dis.eval()
        self.ptc_dis.eval()
        self.clf_dis.eval()

        # encode / decode
        enc_output, dec_output = self.ae(Variable(batch_x), Variable(batch_y))

        # autoencoder loss from reconstruction #重建误差
        loss = self.lambda_ae * ((Variable(batch_x) - dec_output) ** 2).mean()
        self.stats['rec_costs'].append(loss.data[0])

        k_lamda = float(min(self.n_step, self.lambda_final_step)) / self.lambda_final_step  #lamda系数从0到0.0001逐渐增大，增大步长为 1 / self.lamda_final_step
        # encoder loss from the latent discriminator #大概是本文的终极奥义的地方
        lat_dis_preds = self.lat_dis(enc_output)
        flipped = batch_y.add(1) % 2  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 标签翻转
        lat_dis_loss = softmax_cross_entropy(lat_dis_preds, Variable(flipped)) #让编码器去掉属性
        loss = loss + k_lamda * self.lambda_lat_dis * lat_dis_loss

        # decoding with random labels #这里是生成随机的y
        flipped = get_rand_attributes(batch_x.size(0), int(self.n_attr))
        dec_output_flipped = self.ae.decode(enc_output, flipped)

        # autoencoder loss from the patch discriminator # 根ptc_dis_step中产生对抗训练
        ptc_dis_preds = self.ptc_dis(dec_output_flipped)
        y_fake = Variable(torch.FloatTensor(ptc_dis_preds.size()).fill_(self.smooth_label).cuda())
        ptc_dis_loss = F.binary_cross_entropy(ptc_dis_preds, 1 - y_fake) # 让编码器和解码器产生的新数据更加真实
        loss = loss + k_lamda * self.lambda_ptc_dis * ptc_dis_loss

        # autoencoder loss from the classifier discriminator
        clf_dis_preds = self.clf_dis(dec_output_flipped)
        clf_dis_loss = softmax_cross_entropy(clf_dis_preds, flipped) #让编码器和解码器产生的新数据在分类上和y相近
        loss = loss + k_lamda * self.lambda_clf_dis * clf_dis_loss

        # optimize
        self.ae_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.ae.parameters(), self.clip_grad_norm)
        self.ae_optimizer.step()

    """ End the training iteration, print training loss. """
    def printLoss(self, n_iter):
        # average loss
        if len(self.stats['rec_costs']) >= 25:
            mean_loss = [
                ('Latent discriminator', 'lat_dis_costs'),
                ('Patch discriminator', 'ptc_dis_costs'),
                ('Classifier discriminator', 'clf_dis_costs'),
                ('Reconstruction loss', 'rec_costs'),
            ]
            logger.info(('%06i - ' % n_iter) +
                        ' / '.join(['%s : %.5f' % (a, np.mean(self.stats[b])) for a, b in mean_loss]) )
            del self.stats['rec_costs'][:]
            del self.stats['lat_dis_costs'][:]
            del self.stats['ptc_dis_costs'][:]
            del self.stats['clf_dis_costs'][:]
        self.n_step += 1

    """ Reload a previously trained model. """
    def reload_model(self, model, to_reload):
        # reload the model
        to_reload = torch.load(to_reload)
        # copy saved parameters
        for k in model.state_dict().keys():
            model.state_dict()[k].copy_(to_reload.state_dict()[k])

    """ Save the model. """
    def save_model(self, name):
        def save(model, filename):
            path = os.path.join(MODELS_PATH, '%s_%s.pth' % (name, filename))
            logger.info('Saving %s to %s ...' % (filename, path))
            torch.save(model, path)
        save(self.ae, 'ae')
        save(self.lat_dis, 'lat_dis')
        save(self.ptc_dis, 'ptc_dis')
        save(self.clf_dis, 'clf_dis')

    """  Save the best models / periodically save the models. """
    def save_best_periodic(self, to_log):
        if to_log['ae_loss'] < self.best_loss:
            self.best_loss = to_log['ae_loss']
            logger.info('Best reconstruction loss: %.5f' % self.best_loss)
            self.save_model('best_rec')
        if to_log['n_epoch'] % 5 == 0 and to_log['n_epoch'] > 0:
            self.save_model('periodic-%i' % to_log['n_epoch'])


