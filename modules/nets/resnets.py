'''
Function:
	resnet models.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import torchvision
import torch.nn as nn


'''define resnets'''
def ResNets(pretrained=False, backbonename='resnet50', **kwargs):
	if backbonename == 'resnet18':
		model = torchvision.models.resnet18(pretrained=pretrained)
	elif backbonename == 'resnet34':
		model = torchvision.models.resnet34(pretrained=pretrained)
	elif backbonename == 'resnet50':
		model = torchvision.models.resnet50(pretrained=pretrained)
	elif backbonename == 'resnet101':
		model = torchvision.models.resnet101(pretrained=pretrained)
	elif backbonename == 'resnet152':
		model = torchvision.models.resnet152(pretrained=pretrained)
	else:
		raise ValueError('Unsupport ResNets.backbonename <%s>...' % backbonename)
	model = nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
	return model