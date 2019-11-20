'''
Function:
	define the deep dream model.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import torch.nn as nn
from .nets.resnets import ResNets


'''deep dream model'''
class DeepDream(nn.Module):
	def __init__(self, backbonename, **kwargs):
		super(DeepDream, self).__init__()
		if backbonename.find('resnet') != -1:
			self.classifier = ResNets(backbonename=backbonename, pretrained=True)
		else:
			raise ValueError('Unsupport DeepDream.backbonename <%s>...' % backbonename)
	'''forward'''
	def forward(self, x):
		return self.classifier(x)
	'''fix all parameters of the network'''
	def fixParams(self):
		for layer in self.modules():
			for param in layer.parameters():
				param.requires_grad = False