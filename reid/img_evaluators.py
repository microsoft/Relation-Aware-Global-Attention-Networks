from __future__ import absolute_import

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from collections import OrderedDict, Iterable, defaultdict

from .utils import to_torch
from .utils.meters import AverageMeter
from .utils.visualization import get_rank_list
from .evaluation_metrics import cmc, mean_ap

import time
import copy
import numpy as np

# ======================
#  Extracting Features 
# ======================
def inference_feature(model, inputs, feat_type, modules=None):
	model.eval()
	inputs = [to_torch(inputs)]

	## Feature Inference
	if modules is None:
		model_out = model(inputs, training=False)
		if isinstance(feat_type, list):
			outputs = []
			for i in range(len(feat_type)):
				if feat_type[i] == 'feat_':
					outputs.append(model_out[0].data.cpu())
				elif feat_type[i] == 'feat':
					outputs.append(model_out[1].data.cpu())
				else:
					raise ValueError("Cannot support this type of features: {}."
					.format(feat_type))
			return outputs
		elif isinstance(feat_type, str):
			if feat_type == 'feat_':
				outputs = model_out[0]
			elif feat_type == 'feat':
				outputs = model_out[1]
			else:
				raise ValueError("Cannot support this type of features: {}."
					.format(feat_type))
			outputs = outputs.data.cpu()
			return outputs
		else: 
			raise NameError

	## Register forward hook for each module
	outputs = OrderedDict()
	handles = []
	for m in modules:
		outputs[id(m)] = None
		def func(m, i, o): outputs[id(m)] = o.data.cpu()
		handles.append(m.register_forward_hook(func))
	model(inputs)
	for h in handles:
		h.remove()
	return list(outputs.values())

def extract_features(model, data_loader, normlizer, flipper, to_pil, to_tensor,
	feat_type, uv_size=(32, 32), print_freq=1, metric=None):
	model.eval()
	batch_time = AverageMeter()
	data_time = AverageMeter()

	if isinstance(feat_type, list):
		features = {}
		labels = {}
		for feat_name in feat_type:
			features[feat_name] = OrderedDict()
			labels[feat_name] = OrderedDict()
	elif isinstance(feat_type, str):
		features = OrderedDict()
		labels = OrderedDict()
	else:
		raise NameError

	end = time.time()
	for i, (imgs, fnames, pids, _) in enumerate(data_loader):
		data_time.update(time.time() - end)
		in_size = imgs.size()

		## Extract features
		if flipper is not None:
			imgs_flip = copy.deepcopy(imgs)
		else:
			imgs_flip = None
		for j in range(in_size[0]):
			imgs[j, :, :, :] = normlizer(imgs[j, :, :, :])
			if flipper is not None:
				imgs_flip[j, :, :, :] = to_tensor(flipper(to_pil(imgs_flip[j, :, :, :])))
				imgs_flip[j, :, :, :] = normlizer(imgs_flip[j, :, :, :])
		if flipper is not None:
			output_unflip = inference_feature(model, imgs, feat_type)
			output_flip = inference_feature(model, imgs_flip, feat_type)
			outputs = []
			for jj in range(len(output_unflip)):
				outputs.append((output_unflip[jj] + output_flip[jj]) / 2)
		else:
			outputs = inference_feature(model, imgs, feat_type)

		## Save Features
		if isinstance(feat_type, list):
			for ii, feat_name in enumerate(feat_type):
				for fname, output, pid in zip(fnames, outputs[ii], pids):
					features[feat_name][fname] = output
					labels[feat_name][fname] = pid
		elif isinstance(feat_type, str):
			for fname, output, pid in zip(fnames, outputs, pids):
				features[fname] = output
				labels[fname] = pid
		else:
			raise NameError

		batch_time.update(time.time() - end)
		end = time.time()

		if (i + 1) % print_freq == 0:
			print('Extract Features: [{}/{}]\t'
				  'Time {:.3f} ({:.3f})\t'
				  'Data {:.3f} ({:.3f})\t'
				  .format(i + 1, len(data_loader),
						  batch_time.val, batch_time.avg,
						  data_time.val, data_time.avg))

	return features, labels


# =============
#   Evaluator 
# =============
class ImgEvaluator(object):
	def __init__(self, model, file_path, flip_embedding=False):
		super(ImgEvaluator, self).__init__()
		self.model = model
		self.file_path = file_path
		self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		## added for flipping embedding evaluation
		if flip_embedding:
			self.flipper = torchvision.transforms.RandomHorizontalFlip(p=1.0)
			print ('[Info] Flip Embedding is OPENED in evaluation!')
		else:
			self.flipper = None
			print ('[Info] Flip Embedding is CLOSED in evaluation!')
		self.to_pil = torchvision.transforms.ToPILImage()
		self.to_tensor = torchvision.transforms.ToTensor()
	
	def eval_worerank(self, query_loader, gallery_loader, query, gallery, metric, 
		types_list, cmc_topk=(1, 5, 10)):
		query_features_list, _ = extract_features(self.model, query_loader, \
			self.normlizer, self.flipper, self.to_pil, self.to_tensor, types_list)
		gallery_features_list, _ = extract_features(self.model, gallery_loader, \
			self.normlizer, self.flipper, self.to_pil, self.to_tensor, types_list)
		query_features = {}
		gallery_features = {}
		for feat_name in types_list:
			x_q = torch.cat([query_features_list[feat_name][f].unsqueeze(0) for f, _, _ in query], 0)
			x_q = x_q.view(x_q.size(0), -1)
			query_features[feat_name] = x_q

			x_g = torch.cat([gallery_features_list[feat_name][f].unsqueeze(0) for f, _, _ in gallery], 0)
			x_g = x_g.view(x_g.size(0), -1)
			gallery_features[feat_name] = x_g
		
		query_ids = [pid for _, pid, _ in query]
		gallery_ids = [pid for _, pid, _ in gallery]
		query_cams = [cam for _, _, cam in query]
		gallery_cams = [cam for _, _, cam in gallery]

		for feat_name in types_list:
			for dist_type in metric:
				print('Evaluated with "{}" features and "{}" metric:'.format(feat_name, dist_type))
				x = query_features[feat_name]
				y = gallery_features[feat_name]
				m, n = x.size(0), y.size(0)

				# Calculate the distance matrix
				if dist_type == 'euclidean':
					dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
						torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
					dist.addmm_(1, -2, x, y.t())
				elif dist_type == 'cosine':
					x = F.normalize(x, p=2, dim=1)
					y = F.normalize(y, p=2, dim=1)
					dist = 1 - torch.mm(x, y.t())
				else:
					raise NameError

				# Compute mean AP
				mAP = mean_ap(dist, query_ids, gallery_ids, query_cams, gallery_cams)
				print('Mean AP: {:4.1%}'.format(mAP))

				# Compute CMC scores
				cmc_configs = {
					'rank_results': dict(separate_camera_set=False,
									single_gallery_shot=False,
									first_match_break=True)}
				cmc_scores = {name: cmc(dist, query_ids, gallery_ids,
									query_cams, gallery_cams, **params)
							for name, params in cmc_configs.items()}

				print('CMC Scores')
				for k in cmc_topk:
					print('  top-{:<4}{:12.1%}'
						.format(k, cmc_scores['rank_results'][k - 1]))
		return	