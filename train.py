'''
Function:
	train deep dream models.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import cfg
import torch
import argparse
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from modules.utils import *
from torchvision import transforms
from modules.DeepDream import DeepDream


'''parse arguments for training'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Deep Dream')
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for training.', default='resnet50', type=str)
	parser.add_argument('--iscontrolled', dest='iscontrolled', help='whether use an image to control network or not.', action='store_true')
	parser.add_argument('--imagepath', dest='imagepath', help='image path for deep dream.', default='', type=str, required=True)
	parser.add_argument('--controlimagepath', dest='controlimagepath', help='image path for controlling deep dream.', default='', type=str)
	args = parser.parse_args()
	if args.iscontrolled: assert args.controlimagepath
	return args


'''train model'''
def train():
	# parse necessary params
	args = parseArgs()
	# prepare
	checkDir(cfg.SAVEDIR)
	use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	img_ori_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.MEANS, cfg.STDS)])
	img_control_trans = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(cfg.MEANS, cfg.STDS)])
	# define model
	model = DeepDream(backbonename=args.backbonename)
	if use_cuda: model = model.cuda()
	model.train()
	model.fixParams()
	# read image
	img_ori = Image.open(args.imagepath)
	img_ori = img_ori_trans(img_ori).unsqueeze(0)
	features_control = None
	if args.iscontrolled:
		img_control = img_control_trans(Image.open(args.controlimagepath)).unsqueeze(0)
		with torch.no_grad(): features_control = model(img_control.type(FloatTensor))
	# train deep dream
	means = np.array(cfg.MEANS).reshape([3, 1, 1])
	stds = np.array(cfg.STDS).reshape([3, 1, 1])
	img_pyramid = getImagePyramid(img_ori.numpy(), num_octaves=cfg.NUM_OCTAVES, octave_scale=cfg.OCTAVE_SCALE)
	diff = np.zeros_like(img_pyramid[-1])
	# --ascending by resolution
	for idx, octave in enumerate(img_pyramid[::-1]):
		print('START OCTAVE: %s/%s' % (idx+1, len(img_pyramid)))
		if idx > 0:
			h, w = octave.shape[-2:]
			h_ori, w_ori = diff.shape[-2:]
			diff = ndimage.zoom(diff, (1, 1, 1.0*h/h_ori, 1.0*w/w_ori), order=1)
		octave_input = octave + diff
		for iteration in range(cfg.MAX_ITERS):
			print('OCTAVE: %s/%s, ITER: %s/%s...' % (idx+1, len(img_pyramid), iteration+1, cfg.MAX_ITERS))
			shift_x, shift_y = np.random.randint(-cfg.MAX_JITTER, cfg.MAX_JITTER+1, 2)
			octave_input = randomShift(octave_input, shift_x, shift_y)
			octave_input = torch.from_numpy(octave_input).type(FloatTensor)
			octave_input.requires_grad = True
			model.zero_grad()
			output = model(octave_input)
			matched = matchFeatures(output, features_control, FloatTensor)
			output.backward(matched)
			lr = cfg.LEARNING_RATE / np.abs(octave_input.grad.data.cpu().numpy()).mean()
			octave_input.data.add_(octave_input.grad.data * lr)
			if iteration % cfg.SAVE_INTERVAL == 0:
				savepath = os.path.join(cfg.SAVEDIR, '%s_%s.jpg' % (idx, iteration))
				img_reconstruct = reconstructImage(octave_input, shift_x, shift_y, means, stds, savepath)
			else:
				img_reconstruct = reconstructImage(octave_input, shift_x, shift_y, means, stds, savepath, False)
			octave_input = img_reconstruct
		diff = img_reconstruct - octave


'''run'''
if __name__ == '__main__':
	train()