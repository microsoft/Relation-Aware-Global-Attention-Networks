from __future__ import absolute_import

from .rga_model import *

__factory = {
	'resnet50_rga': resnet50_rga,
}


def names():
	return sorted(__factory.keys())

def create(name, *args, **kwargs):
	if name not in __factory:
		raise KeyError("Unknown model:", name)
	return __factory[name](*args, **kwargs)
