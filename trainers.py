# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import print_function, absolute_import
import time
import pdb
import sys
import os
import cv2
import copy
import Image

import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from scipy import misc

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter
from .utils.data.transforms import RandomErasing_UV
from .rgb2uv import warp_rgb_to_uv_gpu

class UVBaseTrainer_RE(object):
	'''
		Supporting for random erasing data augmentation.
	'''
	def __init__(self, model, criterion, summary_writer, prob=0.5, mean=[0.4914, 0.4822, 0.4465], 
		uv_size=(32, 32)):
		super(UVBaseTrainer_RE, self).__init__()
		self.model = model
		self.criterion = criterion
		self.uv_size = uv_size
		self.summary_writer = summary_writer
		self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.eraser = RandomErasing_UV(probability=prob, mean=[0., 0., 0.]) # for [RGB, IUV]

	def train(self, epoch, data_loader, optimizer, print_freq=1):
		self.model.train()

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		precisions = AverageMeter()
		out_path = '/home/v-zhizzh/Project/DenseReID/DensePoseData/infer_out'

		end = time.time()
		for i, inputs in enumerate(data_loader):
			data_time.update(time.time() - end)

			ori_inputs, targets = self._parse_data(inputs)

			in_size = inputs[0].size()
			img_rgb = ori_inputs[:, :, :, :(in_size[3] // 2)]
			img_iuv = ori_inputs[:, :, :, (in_size[3] // 2):]
			del inputs
			del ori_inputs
			uv_inputs = torch.zeros((in_size[0], 24*in_size[1], self.uv_size[0], self.uv_size[1]))

			for j in range(in_size[0]):
				img_rgb[j, :, :, :] = self.normlizer(img_rgb[j, :, :, :])
				img_rgb[j, :, :, :], img_iuv[j, :, :, :] = self.eraser(img_rgb[j, :, :, :], img_iuv[j, :, :, :])
				uv_inputs[j, :, :, :] = warp_rgb_to_uv_gpu(img_rgb[j, :, :, :], img_iuv[j, :, :, :], vflip=True)
			
			inputs = ([img_rgb], uv_inputs)
			loss, all_loss, prec1 = self._forward(inputs, targets)

			losses.update(loss.data, targets.size(0))
			precisions.update(prec1, targets.size(0))
			
			# tensorboard
			if self.summary_writer is not None:
				global_step = epoch * len(data_loader) + i
				self.summary_writer.add_scalar('loss', loss.item(), global_step)
				self.summary_writer.add_scalar('loss_cls_part', all_loss[0], global_step)
				self.summary_writer.add_scalar('loss_tri_part', all_loss[1], global_step)
				self.summary_writer.add_scalar('loss_cls_global', all_loss[2], global_step)
				self.summary_writer.add_scalar('loss_tri_global', all_loss[3], global_step)
				self.summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()

			batch_time.update(time.time() - end)
			end = time.time()

			if (i + 1) % print_freq == 0:
				print('Epoch: [{}][{}/{}]\t'
					  'Time {:.3f} ({:.3f})\t'
					  'Data {:.3f} ({:.3f})\t'
					  'Loss {:.3f} {:.3f} {:.3f} {:.3f}\t'
					  'Prec {:.2%} ({:.2%})\t'
					  .format(epoch, i + 1, len(data_loader),
							  batch_time.val, batch_time.avg,
							  data_time.val, data_time.avg,
							  all_loss[0], all_loss[1], all_loss[2], all_loss[3],
							  precisions.val, precisions.avg))

	def _parse_data(self, inputs):
		raise NotImplementedError

	def _forward(self, inputs, targets):
		raise NotImplementedError

class PGFeatTrainerV2(UVBaseTrainer_RE):
	def _parse_data(self, inputs):
		imgs, _, pids, _ = inputs
		inputs = Variable(imgs)
		targets = Variable(pids.cuda())
		return inputs, targets

	def _forward(self, inputs, targets):
		outputs = self.model(inputs, training=True)

		# part	
		loss_cls_part = []
		for i in range(len(outputs[0][1])):
			loss = 0.5 * self.criterion[0](outputs[0][3][i], targets)
			loss_cls_part.append(loss)
		loss_cls_part = sum(loss_cls_part) / len(loss_cls_part) + \
			self.criterion[0](outputs[2][3], targets)
		loss_tri_part = 1.5 * self.criterion[1](outputs[2][1], targets)
		# global 
		loss_cls_global = 0.5 * self.criterion[0](outputs[0][2], targets) + \
			self.criterion[0](outputs[2][2], targets)
		loss_tri_global = 1.5 * self.criterion[1](outputs[2][0], targets)
		
		loss = loss_cls_part + loss_tri_part + loss_cls_global + loss_tri_global
		losses = [loss_cls_part, loss_tri_part, loss_cls_global, loss_tri_global]  
		prec, = accuracy(outputs[-1][2].data, targets.data)
		prec = prec[0]
		return loss, losses, prec
