from __future__ import print_function, absolute_import
import time
import sys
import os

import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from scipy import misc

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils.data.transforms import RandomErasing

class BaseTrainer(object):
	def __init__(self, model, criterion, summary_writer, prob=0.5, mean=[0.4914, 0.4822, 0.4465]):
		super(BaseTrainer, self).__init__()
		self.model = model
		self.criterion = criterion
		self.summary_writer = summary_writer
		self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.eraser = RandomErasing(probability=prob, mean=[0., 0., 0.])

	def train(self, epoch, data_loader, optimizer, random_erasing, empty_cache=False, print_freq=1):
		self.model.train()

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		precisions = AverageMeter()

		end = time.time()
		for i, inputs in enumerate(data_loader):
			data_time.update(time.time() - end)

			ori_inputs, targets = self._parse_data(inputs)
			in_size = inputs[0].size()
			for j in range(in_size[0]):
				ori_inputs[0][j, :, :, :] = self.normlizer(ori_inputs[0][j, :, :, :])
				if random_erasing:
					ori_inputs[0][j, :, :, :] = self.eraser(ori_inputs[0][j, :, :, :])
			loss, all_loss, prec1 = self._forward(ori_inputs, targets)

			losses.update(loss.data, targets.size(0))
			precisions.update(prec1, targets.size(0))

			# tensorboard
			if self.summary_writer is not None:
				global_step = epoch * len(data_loader) + i
				self.summary_writer.add_scalar('loss', loss.item(), global_step)
				self.summary_writer.add_scalar('loss_cls', all_loss[0], global_step)
				self.summary_writer.add_scalar('loss_tri', all_loss[1], global_step)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if empty_cache:
				torch.cuda.empty_cache()

			batch_time.update(time.time() - end)
			end = time.time()

			if (i + 1) % print_freq == 0:
				print('Epoch: [{}][{}/{}]\t'
					'Time {:.3f} ({:.3f})\t'
					'Data {:.3f} ({:.3f})\t'
					'Loss {:.3f} {:.3f} \t'
					'Prec {:.2%} ({:.2%})\t'
					.format(epoch, i + 1, len(data_loader),
							batch_time.val, batch_time.avg,
							data_time.val, data_time.avg,
							all_loss[0], all_loss[1],
							precisions.val, precisions.avg))
				
	def _parse_data(self, inputs):
		raise NotImplementedError

	def _forward(self, inputs, targets):
		raise NotImplementedError


class ImgTrainer(BaseTrainer):
	def _parse_data(self, inputs):
		imgs, _, pids, _ = inputs
		inputs = [Variable(imgs)]
		targets = Variable(pids.cuda())
		return inputs, targets

	def _forward(self, inputs, targets):
		outputs = self.model(inputs, training=True)
		
		loss_cls = self.criterion[0](outputs[2], targets)
		loss_tri = self.criterion[1](outputs[0], targets)
		
		loss = loss_cls + loss_tri
		losses = [loss_cls, loss_tri]  
		prec, = accuracy(outputs[2].data, targets.data)
		prec = prec[0]
		return loss, losses, prec

