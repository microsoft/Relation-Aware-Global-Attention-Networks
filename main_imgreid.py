# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime

from reid import data_manager
from reid import models
from reid.img_trainers import ImgTrainer
from reid.img_evaluators import ImgEvaluator
from reid.loss.loss_set import TripletHardLoss, CrossEntropyLabelSmoothLoss
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.lr_scheduler import LRScheduler

def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
			 workers, combine_trainval, eval_rerank=False):
	## Datasets
	if name == 'cuhk03labeled':
		dataset_name = 'cuhk03'
		dataset = data_manager.init_imgreid_dataset(
			root=data_dir, name=dataset_name, split_id=split_id,
			cuhk03_labeled=True, cuhk03_classic_split=False,
		)
		dataset.images_dir = osp.join(data_dir, '/CUHK03_New/images_labeled/')
	elif name == 'cuhk03detected':
		dataset_name = 'cuhk03'
		dataset = data_manager.init_imgreid_dataset(
			root=data_dir, name=dataset_name, split_id=split_id,
			cuhk03_labeled=False, cuhk03_classic_split=False,
		)
		dataset.images_dir = osp.join(data_dir, '/CUHK03_New/images_detected/')
	## Num. of training IDs
	num_classes = dataset.num_train_pids

	train_transformer = T.Compose([
		T.Random2DTranslation(height, width),
		T.RandomHorizontalFlip(),
		T.ToTensor(),
	])

	test_transformer = T.Compose([
		T.RectScale(height, width),
		T.ToTensor(),
	])

	train_loader = DataLoader(
		Preprocessor(dataset.train, root=dataset.images_dir, transform=train_transformer),
		batch_size=batch_size, num_workers=workers,
		sampler=RandomIdentitySampler(dataset.train, num_instances),
		pin_memory=True, drop_last=True)

	query_loader = DataLoader(
		Preprocessor(dataset.query, root=dataset.images_dir, transform=test_transformer),
		batch_size=batch_size, num_workers=workers,
		shuffle=False, pin_memory=True)

	gallery_loader = DataLoader(
		Preprocessor(dataset.gallery, root=dataset.images_dir, transform=test_transformer),
		batch_size=batch_size, num_workers=workers,
		shuffle=False, pin_memory=True)

	return dataset, num_classes, train_loader, query_loader, gallery_loader


def main(args):
	## Set the seeds
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	cudnn.benchmark = True

	## Create data loaders
	assert args.num_instances > 1, "num_instances should be greater than 1"
	assert args.batch_size % args.num_instances == 0, \
		'num_instances should divide batch_size'
	if args.height is None or args.width is None:
		args.height, args.width = (144, 56) if args.arch == 'inception' else \
								  (256, 128)

	dataset, num_classes, train_loader, query_loader, gallery_loader = \
		get_data(args.dataset, args.split, args.data_dir, args.height,
				args.width, args.batch_size, args.num_instances, args.workers,
				args.combine_trainval, args.rerank)
	
	## Summary Writer
	if not args.evaluate:
		TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
		summary_writer = SummaryWriter(osp.join(args.logs_dir, 'tensorboard_log'+TIMESTAMP))
	else:
		summary_writer = None
		
	## Create model
	model = models.create(args.arch, pretrained=True, num_feat=args.features, 
				height=args.height, width=args.width, dropout=args.dropout, 
				num_classes=num_classes, branch_name=args.branch_name)

	## Load from checkpoint
	start_epoch = best_top1 = 0
	if args.resume:
		checkpoint = load_checkpoint(args.resume)
		model.load_state_dict(checkpoint['state_dict'])
		start_epoch = checkpoint['epoch']
		best_top1 = checkpoint['best_top1']
		print("=> Start epoch {}  best top1 {:.1%}"
			  .format(start_epoch, best_top1))
	model = nn.DataParallel(model).cuda()

	## Evaluator
	evaluator = ImgEvaluator(model, file_path=args.logs_dir)
	# test/evaluate the model
	if args.evaluate:
		feats_list = ['feat_', 'feat']
		evaluator.eval_worerank(query_loader, gallery_loader, dataset.query, dataset.gallery, 
			metric=['cosine'], 
			types_list=feats_list)
		return

	## Criterion
	criterion_cls = CrossEntropyLabelSmoothLoss(dataset.num_train_pids).cuda()
	criterion_tri = TripletHardLoss(margin=args.margin)
	criterion = [criterion_cls, criterion_tri]

	## Trainer
	trainer = ImgTrainer(model, criterion, summary_writer)

	## Optimizer
	if hasattr(model.module, 'backbone'):
		base_param_ids = set(map(id, model.module.backbone.parameters()))
		new_params = [p for p in model.parameters() if
					  id(p) not in base_param_ids]   
		param_groups = [
			{'params': filter(lambda p: p.requires_grad,model.module.backbone.parameters()), 'lr_mult': 1.0},
			{'params': filter(lambda p: p.requires_grad,new_params), 'lr_mult': 1.0}]
	else:
		param_groups = model.parameters()
	if args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(param_groups, lr=args.lr,
									momentum=args.momentum,
									weight_decay=args.weight_decay,
									nesterov=True)
	elif args.optimizer == 'adam':
		optimizer = torch.optim.Adam(
			param_groups, lr=args.lr, weight_decay=args.weight_decay
		)
	else:
		raise NameError
	if args.resume and checkpoint.has_key('optimizer'):
		optimizer.load_state_dict(checkpoint['optimizer'])

	## Learning rate scheduler
	lr_scheduler = LRScheduler(base_lr=0.0008, step=[80, 120, 160, 200, 240, 280, 320, 360],
							factor=0.5, warmup_epoch=20,
							warmup_begin_lr=0.000008)
	
	## Start training
	for epoch in range(start_epoch, args.epochs):
		lr = lr_scheduler.update(epoch)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		print('[Info] Epoch [{}] learning rate update to {:.3e}'.format(epoch, lr))
		trainer.train(epoch, train_loader, optimizer, random_erasing=args.random_erasing, empty_cache=args.empty_cache)
		if (epoch + 1) % 40 == 0 and (epoch + 1) >= args.start_save:
			is_best = False
			save_checkpoint({
				'state_dict': model.module.state_dict(),
				'epoch': epoch + 1,
				'best_top1': best_top1,
				'optimizer': optimizer.state_dict(),
			}, epoch + 1, is_best, save_interval=1, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))


if __name__ == '__main__':
	def str2bool(v):
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Unsupported value encountered.')
			
	parser = argparse.ArgumentParser(description="Softmax loss classification")
	# data
	parser.add_argument('-d', '--dataset', type=str, default='cuhk03')
	parser.add_argument('-b', '--batch-size', type=int, default=256)
	parser.add_argument('-j', '--workers', type=int, default=4)
	parser.add_argument('--split', type=int, default=0)
	parser.add_argument('--height', type=int,
						help="input height, default: 256 for resnet*, "
							 "144 for inception")
	parser.add_argument('--width', type=int,
						help="input width, default: 128 for resnet*, "
							 "56 for inception")
	parser.add_argument('--combine-trainval', action='store_true',
						help="train and val sets together for training, "
							 "val set alone for validation")
	parser.add_argument('--num-instances', type=int, default=4,
						help="each minibatch consist of "
							 "(batch_size // num_instances) identities, and "
							 "each identity has num_instances instances, "
							 "default: 4")
	# model
	parser.add_argument('-a', '--arch', type=str, default='resnet50',
						choices=models.names())
	parser.add_argument('--features', type=int, default=2048)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--branch_name', type=str, default='rgasc')
	parser.add_argument('--use_rgb', type=str2bool, default=True)
	parser.add_argument('--use_bn', type=str2bool, default=True)
	# loss
	parser.add_argument('--margin', type=float, default=0.3,
						help="margin of the triplet loss, default: 0.3")
	# optimizer
	parser.add_argument('-opt', '--optimizer', type=str, default='sgd')
	parser.add_argument('--lr', type=float, default=0.1,
						help="learning rate of new parameters, for pretrained "
							 "parameters it is 10 times smaller than this")
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--weight-decay', type=float, default=5e-4)
	# training configs
	parser.add_argument('--num_gpu', type=int, default=4)
	parser.add_argument('--resume', type=str, default='', metavar='PATH')
	parser.add_argument('--evaluate', action='store_true',
						help="evaluation only")
	parser.add_argument('--rerank', action='store_true',
						help="evaluation with re-ranking")
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--start_save', type=int, default=0,
						help="start saving checkpoints after specific epoch")
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--print-freq', type=int, default=1)
	parser.add_argument('--empty_cache', type=str2bool, default=False)
	parser.add_argument('--random_erasing', type=str2bool, default=True)
	# metric learning
	parser.add_argument('--dist-metric', type=str, default='euclidean',
						choices=['euclidean', 'kissme'])
	# misc
	working_dir = osp.dirname(osp.abspath(__file__))
	parser.add_argument('--data-dir', type=str, metavar='PATH',
						default='/home/datasets')
	parser.add_argument('--logs-dir', type=str, metavar='PATH',
						default=osp.join(working_dir, 'logs'))
	parser.add_argument('--logs-file', type=str, metavar='PATH', 
						default='log.txt')
	main(parser.parse_args())