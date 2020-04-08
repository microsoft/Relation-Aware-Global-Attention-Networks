from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def normalize(x, axis=-1):
	"""Normalizing to unit length along the specified dimension.
	Args:
	  x: pytorch Variable
	Returns:
	  x: pytorch Variable, same shape as input
	"""
	x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
	return x

def euclidean_dist(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	Returns:
	  dist: pytorch Variable, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	"""
	x_normed = F.normalize(x, p=2, dim=1)
	y_normed = F.normalize(y, p=2, dim=1)
	return 1 - torch.mm(x_normed, y_normed.t())

def cosine_similarity(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	"""
	x_normed = F.normalize(x, p=2, dim=1)
	y_normed = F.normalize(y, p=2, dim=1)
	return torch.mm(x_normed, y_normed.t())


def hard_example_mining(dist_mat, labels, return_inds=False):
	"""For each anchor, find the hardest positive and negative sample.
	Args:
	  dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
	  labels: pytorch LongTensor, with shape [N]
	  return_inds: whether to return the indices. Save time if `False`(?)
	Returns:
	  dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
	  dist_an: pytorch Variable, distance(anchor, negative); shape [N]
	  p_inds: pytorch LongTensor, with shape [N];
		indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
	  n_inds: pytorch LongTensor, with shape [N];
		indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
	NOTE: Only consider the case in which all labels have same num of samples,
	  thus we can cope with all anchors in parallel.
	"""
	assert len(dist_mat.size()) == 2
	assert dist_mat.size(0) == dist_mat.size(1)
	N = dist_mat.size(0)

	is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
	is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

	dist_ap, relative_p_inds = torch.max(
		dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
	dist_an, relative_n_inds = torch.min(
		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
	
	dist_ap = dist_ap.squeeze(1)
	dist_an = dist_an.squeeze(1)

	if return_inds:
		ind = (labels.new().resize_as_(labels)
			   .copy_(torch.arange(0, N).long())
			   .unsqueeze(0).expand(N, N))
		p_inds = torch.gather(
			ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
		n_inds = torch.gather(
			ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
		p_inds = p_inds.squeeze(1)
		n_inds = n_inds.squeeze(1)
		return dist_ap, dist_an, p_inds, n_inds

	return dist_ap, dist_an


# ==============
#  Triplet Loss 
# ==============
class TripletHardLoss(object):
	"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
	Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
	Loss for Person Re-Identification'."""
	def __init__(self, margin=None, metric="euclidean"):
		self.margin = margin
		self.metric = metric
		if margin is not None:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def __call__(self, global_feat, labels, normalize_feature=False):
		if normalize_feature:
			global_feat = normalize(global_feat, axis=-1)

		if self.metric == "euclidean":
			dist_mat = euclidean_dist(global_feat, global_feat)
		elif self.metric == "cosine":
			dist_mat = cosine_dist(global_feat, global_feat)
		else:
			raise NameError

		dist_ap, dist_an = hard_example_mining(
			dist_mat, labels)
		y = dist_an.new().resize_as_(dist_an).fill_(1)
		
		if self.margin is not None:
			loss = self.ranking_loss(dist_an, dist_ap, y)
		else:
			loss = self.ranking_loss(dist_an - dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss

# ======================
#  Classification Loss 
# ======================
class CrossEntropyLabelSmoothLoss(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.
	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.
	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""
	def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
		super(CrossEntropyLabelSmoothLoss, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
		if self.use_gpu: targets = targets.cuda()
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss
