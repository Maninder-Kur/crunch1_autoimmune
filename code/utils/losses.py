import warnings
import h5py
import csv
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from scipy.stats import pearsonr, ConstantInputWarning
from datetime import datetime
import logging
import argparse
import json
import pandas as pd
from scipy import sparse
import torchsort
from scipy.spatial.distance import cdist
from compute_metrics import spearmanrr



torch.manual_seed(42)

def custom_loss_l1_l2_pred_cosine(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    l2_loss = torch.mean(torch.pow(y_pred, exponent=2))  #L2 regularized to encourage elsatic net

    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    return mse + lambda_reg * l1_loss + lambda_reg * l2_loss + lambda_reg * mean_cosine_dist


def custom_loss_l1_pred_cosine(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_l1=0.2):
    mse = MSELoss()(y_pred, y_true)
    
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity

    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    return mse + lambda_l1 * l1_loss + lambda_reg * mean_cosine_dist


def custom_loss_l1_pred_l2_weights_cosine(y_true, y_pred, neighbors_gene_expression, model, lambda_l1=0.1, lambda_l2=0.1, lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = lambda_l1 * torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    l2_loss = lambda_l2 * sum(
        torch.sum(param ** 2) 
        for name, param in model.named_parameters() 
        if param.requires_grad and "bias" not in name
    )

    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    return mse + l1_loss + l2_loss + lambda_reg * mean_cosine_dist

def custom_loss_l1_spearman(y_true, y_pred,  lambda_l1=0.1,  lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)
    # Flatten the neighbors for cosine similarity calculation
    # neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    spearman_loss = 1-spearmanrr(y_pred, y_true, regularization_strength=1.0)


    return mse + lambda_l1 * l1_loss + spearman_loss


def custom_loss_pred_zeros(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_zero=2.0, lambda_bce=0.5):
    """
    Custom loss function to handle imbalanced zeros and non-zeros in predictions.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth tensor of shape (batch_size, output_size).
    y_pred : torch.Tensor
        Predicted tensor of shape (batch_size, output_size).
    neighbors_gene_expression : torch.Tensor
        Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
    lambda_reg : float
        Weight for cosine similarity regularization.
    lambda_zero : float
        Weight for zero-penalty regularization.
    lambda_bce : float
        Weight for BCE loss for zero prediction accuracy.
    """
    # Mean Squared Error (MSE) Loss for all predictions
    mse = torch.nn.functional.mse_loss(y_pred, y_true)

    # Zero-Penalty Regularization: Penalize non-zero predictions for zero ground truth
    zero_mask = (y_true == 0).float()
    zero_penalty = torch.mean((y_pred * zero_mask) ** 2)

    # Weighted MSE for Non-Zero Targets
    non_zero_mask = (y_true != 0).float()
    weighted_mse = torch.mean(non_zero_mask * (y_pred - y_true) ** 2)

    # Cosine Similarity Regularization
    batch_size = y_pred.size(0)
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)

    # Binary Cross-Entropy Loss for Zero Prediction
    zero_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, zero_mask)

    # Total Loss
    total_loss = (
        weighted_mse
        + lambda_zero * zero_penalty
        + lambda_bce * zero_bce_loss
        + lambda_reg * mean_cosine_dist
    )

    return total_loss


def custom_spearmanr_old(pred, target, alpha=2.0, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    # Soft rank transformation
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    # Apply non-zero weighting
    non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
    # print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return 1 - weighted_corr  # Convert correlation to loss


def custom_spearmanr_with_mse(pred, target, alpha=2.0, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    # Soft rank transformation
    mse = MSELoss()(pred, target)
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    # Apply non-zero weighting
    non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
    print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return mse + 1 - weighted_corr  # Convert correlation to loss


def spearmanrr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()


def spearmanrr_feature_wise(pred, target, **kw):
    pred_rank = torchsort.soft_rank(pred, **kw)
    target_rank = torchsort.soft_rank(target, **kw)

    # Normalize each feature separately
    pred_rank = pred_rank - pred_rank.mean(dim=0, keepdim=True)
    pred_rank = pred_rank / (pred_rank.norm(dim=0, keepdim=True) + 1e-6)  # Avoid division by zero

    target_rank = target_rank - target_rank.mean(dim=0, keepdim=True)
    target_rank = target_rank / (target_rank.norm(dim=0, keepdim=True) + 1e-6)

    # Compute Spearman correlation per feature (reduce across batch dimension)
    spearman_corr = (pred_rank * target_rank).sum(dim=0)  # (460,)

    # Return the mean correlation across all features
    return 1-spearman_corr.mean()  # Scalar loss value
