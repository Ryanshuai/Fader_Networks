# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
from logging import getLogger

from .utils import get_rand_attributes, change_ith_attribute_to_j
from torch.autograd import Variable

logger = getLogger()

"""  Evaluator initialization. """
class Evaluator(object):
    def __init__(self, ae, lat_dis, ptc_dis, clf_dis, data, params):

        # data / parameters
        self.data = data
        self.params = params

        # modules
        self.ae = ae
        self.lat_dis = lat_dis
        self.ptc_dis = ptc_dis
        self.clf_dis = clf_dis

    """  Compute the AE reconstruction loss. """
    def eval_reconstruction_loss(self):
        self.ae.eval()
        costs = []
        for i, batch in enumerate(self.data):
            try:
                batch_x, batch_y = Variable(batch[0].cuda(), volatile=True), Variable(batch[1].cuda(), volatile=True)
                _, dec_output = self.ae(batch_x, batch_y)
                costs.append(((dec_output - batch_x) ** 2).mean().data[0])
            except:
                pass
        return np.mean(costs)

    """ Compute the latent discriminator prediction accuracy. """
    def eval_lat_dis_accuracy(self):
        self.ae.eval()
        self.lat_dis.eval()

        bs_accu = []
        for i, batch in enumerate(self.data):
            try:
                batch_x = Variable(batch[0].cuda(), volatile=True)
                enc_output = self.ae.encode(batch_x)
                preds = self.lat_dis(enc_output).data.cpu()
                preds = np.argmax(preds.numpy(),axis=-1) # [bs , n_attr]
                targets = np.argmax(batch[1].cpu().numpy(), axis=-1)  # [bs , n_attr]
                bs_accu.append(np.mean(preds == targets,axis=0)) #每个属性一个均值, [bs , n_attr]->[1,n_attr] .bs_accu是np.array的list, np.array的size为【1，n_attr】
            except:
                pass
        return np.mean(bs_accu, axis=0) #size:[1,n_attr] 每个属性一个均值

    """ Compute the patch discriminator prediction accuracy. """
    def eval_ptc_dis_accuracy(self):
        params = self.params
        self.ae.eval()
        self.ptc_dis.eval()
        bs = params.batch_size

        real_preds = []
        fake_preds = []
        for i, batch in enumerate(self.data):
            try:
                batch_x= Variable(batch[0].cuda(), volatile=True)
                flipped = get_rand_attributes(bs, params.n_attr)
                flipped = flipped.view(bs, -1)
                _, dec_output = self.ae(batch_x, flipped)
                # predictions
                real_preds.extend(self.ptc_dis(batch_x).data.tolist())
                fake_preds.extend(self.ptc_dis(dec_output).data.tolist())
            except:
                pass
        return real_preds, fake_preds

    """ Compute the classifier discriminator prediction accuracy. """
    def eval_clf_dis_accuracy(self):

        self.ae.eval()
        self.clf_dis.eval()

        bs_accu = []
        for i, batch in enumerate(self.data):
            try:
                batch_x, batch_y = Variable(batch[0].cuda(), volatile=True), Variable(batch[1].cuda(), volatile=True)
                enc_output = self.ae.encode(batch_x)
                # flip all attributes one by one
                for i in range(self.params.n_attr):
                    for value in range(2):
                        flipped = change_ith_attribute_to_j(batch_y, i , value)
                        dec_output = self.ae.decode(enc_output, flipped)
                        # classify
                        clf_dis_preds = self.clf_dis(dec_output)
                        preds = np.argmax(clf_dis_preds.numpy(), axis=-1)  # [bs , n_attr]
                        targets = np.argmax(batch[1].cpu().numpy(), axis=-1)  # [bs , n_attr]
                        bs_accu.append(np.mean(preds == targets, axis=0))  # 每个属性一个均值, [bs , n_attr]->[1,n_attr] .bs_accu是np.array的list, np.array的size为【1，n_attr】
            except:
                pass
        return np.mean(bs_accu, axis=0)  # size:[1,n_attr] 每个属性一个均值

    """ print accuracies. """
    def print_accuracies(self, values):
        for name, value in values:
            logger.info('{:<20}: {:>6}'.format(name, '%.3f%%' % (100 * value)))
        logger.info('')

    """  Evaluate all models / log evaluation results. """
    def evaluate(self, n_epoch):
        params = self.params
        logger.info('')

        # reconstruction loss
        ae_loss = self.eval_reconstruction_loss()
        logger.info('Autoencoder loss: %.5f' % ae_loss)

        # latent discriminator accuracy
        lat_dis = []

        lat_dis_accu = self.eval_lat_dis_accuracy()
        lat_dis.append(('lat_dis_accu', np.mean(lat_dis_accu))) # 所有属性的均值
        for accu, name in zip(lat_dis_accu, params.attr):
            lat_dis.append(('lat_dis_accu_%s' % name, accu)) # 每个属性的均值
        logger.info('Latent discriminator accuracy:')
        self.print_accuracies(lat_dis)

        # patch discriminator accuracy
        ptc_dis = []
        real_preds, fake_preds = self.eval_ptc_dis_accuracy()
        accu_real = (np.array(real_preds).astype(np.float32) >= 0.5).mean()
        accu_fake = (np.array(fake_preds).astype(np.float32) <= 0.5).mean()
        ptc_dis.append(('ptc_dis_preds_real', np.mean(real_preds)))
        ptc_dis.append(('ptc_dis_preds_fake', np.mean(fake_preds)))
        ptc_dis.append(('ptc_dis_accu_real', accu_real))
        ptc_dis.append(('ptc_dis_accu_fake', accu_fake))
        ptc_dis.append(('ptc_dis_accu', (accu_real + accu_fake) / 2))
        logger.info('Patch discriminator accuracy:')
        self.print_accuracies(ptc_dis)

        # classifier discriminator accuracy
        clf_dis = []
        clf_dis_accu = self.eval_clf_dis_accuracy()
        k = 0
        clf_dis += [('clf_dis_accu', np.mean(clf_dis_accu))]
        for accu, name in zip(lat_dis_accu, params.attr):
            clf_dis.append(('clf_dis_accu_%s' % name, accu))
        logger.info('Classifier discriminator accuracy:')
        self.print_accuracies(clf_dis)

        # JSON log
        to_log = dict([
            ('n_epoch', n_epoch),
            ('ae_loss', ae_loss)
        ] + lat_dis + ptc_dis + clf_dis)# + log_clf)
        logger.debug("__log__:%s" % json.dumps(to_log))

        return to_log


