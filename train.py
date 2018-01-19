#! /usr/bin/env python

import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dataset
import models
import config

# Type Parameters
ltype = config.ltype
ftype = config.ftype
# Training Parameters
learning_rate = config.learning_rate

def parameters(*argv):
    params = []
    for model in argv:
        params += list(model.parameters())

    return params

def print_score(batches, step):
    batch_loss = 0. # hit count
    for i, batch in enumerate(batches):
        user_batch, context_batch, target_batch = zip(*batch) 
        batch_loss += run(user_batch, context_batch, target_batch, step=step)
    print("Validation Error :", batch_loss/i, time.ctime())

##############################################################################################
def run(user, context, target, step):

    optimizer.zero_grad()

    user = Variable(torch.from_numpy(np.asarray(user)).type(ltype))
    context = Variable(torch.from_numpy(np.asarray(context)).type(ltype))

    # POI2VEC
    loss = p2v_model(user, context, target)

    loss.backward()
    optimizer.step()
    gc.collect()

    return loss.data.cpu().numpy()[0]

##############################################################################################
##############################################################################################
if __name__ == "__main__":

    # Data Preparation
    data = dataset.Data()
    poi_cnt, user_cnt = data.load()

    # Model Preparation
    p2v_model = models.POI2VEC(poi_cnt, user_cnt, data.id2route, data.id2lr, data.id2prob).cuda()
    loss_model = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(parameters(p2v_model), lr=config.learning_rate, momentum=config.momentum)

    for i in xrange(config.num_epochs):
        # Training
        batch_loss = 0.
        train_batches = data.train_batch_iter(config.batch_size)
        for j, train_batch in enumerate(train_batches):
            user_batch, context_batch, target_batch = zip(*train_batch) 
            batch_loss += run(user_batch, context_batch, target_batch, step=1)
            if (j+1) % 1000 == 0:
                print("batch #{:d}: ".format(j+1), "batch_loss :", batch_loss/j, time.ctime())

        # Validation 
        if (i+1) % config.evaluate_every == 0:
            print("==================================================================================")
            print("Evaluation at epoch #{:d}: ".format(i+1))
            p2v_model.eval()
            valid_batches = data.valid_batch_iter(config.batch_size)
            print_score(valid_batches, step=2)
            p2v_model.train()

# Test
print("==================================================================================")
print("Testing")
p2v_model.eval()
test_batches = data.test_batch_iter(config.batch_size)
print_score(test_batches, step=2)
