import torch

# Parameters
# ==================================================
ltype = torch.cuda.LongTensor
ftype = torch.cuda.FloatTensor

# Model Hyperparameters
feat_dim = 200
route_depth = 16
route_count = 4
context_len = 32

# Weight init
weight_m = 0
weight_v = 0.005

# Training Parameters
batch_size = 128
num_epochs = 30
learning_rate = 0.005
momentum = 0.9
evaluate_every = 3
