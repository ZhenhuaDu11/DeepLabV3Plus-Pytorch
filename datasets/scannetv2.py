import os.path
import glob
import random
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import cv2

nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)

class Scannetv2Dataset():

	def __init__(self, data_root, phase):
		self.phase=phase
		self.root = os.path.join(data_root,'ScanNet/scannet_frames/scannet_frames_25k')
		self.num_labels = 41
		self.ignore_label = 0
		self.class_weights = None
		with open('./datasets/scannet/scannetv2_weigths.txt') as f:
			weights = f.readlines()
		self.class_weights = torch.from_numpy(np.array([float(x.strip()) for x in weights]))
		self.class_weights = self.class_weights.type(torch.FloatTensor)

		with open('./datasets/scannet/scannetv2_{}.txt'.format(phase)) as f:
			scans = f.readlines()
		self.scans = [x.strip() for x in scans]

		self.rgb_frames = []
		self.depth_frames = []
		self.masks = []

		self.total_frames = 0
		for scan in self.scans:
			rgb_frames = glob.glob("{}/{}/color/*.jpg".format(self.root, scan))
			depth_frames = glob.glob("{}/{}/depth/*.png".format(self.root, scan))
			masks = glob.glob("{}/{}/label/*.png".format(self.root, scan))
			if len(rgb_frames) == len(depth_frames):
				rgb_frames.sort()
				depth_frames.sort()
				masks.sort()
				self.total_frames += len(rgb_frames)
				self.rgb_frames.extend(rgb_frames)
				self.depth_frames.extend(depth_frames)
				self.masks.extend(masks)

	def __getitem__(self, index):

		size = (640,480)
		rgb_image = np.array(Image.open(self.rgb_frames[index]))
		rgb_image = cv2.resize(rgb_image, size, interpolation=cv2.INTER_LINEAR)
		depth_image = np.array(Image.open(self.depth_frames[index]))
		depth_image = cv2.resize(depth_image, size,interpolation=cv2.INTER_NEAREST).astype(np.float)
		depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255
		depth_image = depth_image.astype(np.uint8)
		mask_fullsize = []
		if self.masks:
			mask = np.array(Image.open(self.masks[index]))
			mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
			if self.phase == "val":
				mask_fullsize = np.array(Image.open(self.masks[index]))
		else: # test phase
			mask = np.zeros((480, 640), dtype=int)
			mask_fullsize = np.zeros((968, 1296), dtype=int)

		rgb_image = transforms.ToTensor()(rgb_image)
		rgb_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(rgb_image)
		rgb_image = rgb_image.type(torch.FloatTensor)
		depth_image = transforms.ToTensor()(depth_image[:, :, np.newaxis])
		depth_image = depth_image.type(torch.FloatTensor)

		mask = torch.from_numpy(mask)
		mask = mask.type(torch.LongTensor)

		return rgb_image, mask
		# return {'rgb_image': rgb_image, 'depth_image': depth_image,
		#         'mask': mask, 'mask_fullsize': mask_fullsize,
		# 		'path': self.depth_frames[index].split('/')[-1], 'scan':self.depth_frames[index].split('/')[-3]}

	def __len__(self):
		return self.total_frames
    
	def load_data(self):
		return self
	
	@classmethod
	def decode_target(cls,target):
		colour_map_np=nyu40_colour_code
		vis_label = colour_map_np[target]
		
		return vis_label
	
	def name(self):
		return 'Scannetv2'
