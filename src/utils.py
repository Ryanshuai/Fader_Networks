# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
import torch
from torch.autograd import Variable
from logging import getLogger
import logging
import time
from datetime import timedelta



import torch.nn.functional as F

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

logger = getLogger()

""" Compute softmax loss. """
def softmax_cross_entropy(pred, label,axis = -1):
    bs = label.size()[0]
    label_ = label.view(bs, -1)
    pred_ = F.log_softmax(pred, axis).view(bs, -1)
    loss = -torch.dot(label_, pred_) / bs
    return loss

def get_rand_attributes(BS, y_dim):
    y = torch.LongTensor(BS, y_dim).random_(2) #生成一个[BS, y_dim]的随机矩阵，随机值0，1
    y = one_hot(y, 2)
    return Variable(y.cuda())

def change_ith_attribute_to_j(attributes, i, j): #j代表位置
    attributes = attributes
    attributes[:,i] = j # i 设为 j.  example attributes[0,1] i = 0, j =1 -> attributes = [1,1]
    attributes[:, 1-i] = 1-j # 1-i 设为 1-j, 1-i = 1, 1-j = 0->-> attributes = [1,0]
    return attributes

def one_hot(label, depth, axis = -1): #by yanshuai
    label = torch.LongTensor(label)
    label = label.unsqueeze(axis) #[BS]->[BS,1] #[BS,y]->[BS,y,1]
    label_size = list(label.size())
    len_dim = len(label_size)
    label_size[axis] = depth
    label_expend = label.expand(label_size)
    zero_label_expend = torch.zeros_like(label_expend)
    one_hot = zero_label_expend.scatter_(len_dim + axis, label, 1)
    one_hot = one_hot.type(torch.FloatTensor)
    return one_hot


def initialize_exp(params):
    """
    Experiment initialization.
    """
    # dump parameters
    params.dump_path = get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items()))) # 打印params参数值
    return logger

