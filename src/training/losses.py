import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstruction_loss(prediction, truth):
    return F.mse_loss(prediction, truth)

def sparsity_loss(_lambda:float, features):
    return _lambda*F.l1_loss(features, torch.zeros_like(features))

def compute_loss(_lambda, prediction, truth, features):
    reconstruction_loss_result = reconstruction_loss(prediction, truth)
    sparsity_loss_result = sparsity_loss(_lambda, features)
    return reconstruction_loss_result + sparsity_loss_result