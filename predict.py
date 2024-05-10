from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import cv2

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics
from datasets import scannetv2

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def setup_seed(seed):
    print(f'random seed :{seed}')

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def get_argparser(scene, mode):
    parser = argparse.ArgumentParser()
    data_dir = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
    input = os.path.join(data_dir, scene, 'image', mode)
    output = os.path.join(data_dir, scene, 'semantic', mode)
    # Datset Options
    parser.add_argument("--input", type=str, required=False,
                        default=input)
    parser.add_argument("--dataset", type=str, default='scannet',
                        choices=['voc', 'cityscapes', 'scannet'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=output,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default='checkpoints/save/best_deeplabv3plus_resnet101_scannet_os16.pth', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main(scene, mode):
    opts = get_argparser(scene, mode).parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'scannet':
        opts.num_classes = 40
        decode_fn = scannetv2.Scannetv2Dataset.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
        os.makedirs(os.path.join(opts.save_val_results_to,'deeplab_logits'), exist_ok=True)
        os.makedirs(os.path.join(opts.save_val_results_to,'deeplab'), exist_ok=True)
        os.makedirs(os.path.join(opts.save_val_results_to,'deeplab_vis'), exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files, desc='predicting semantic'):
            img_name = os.path.basename(img_path)
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            logits = model(img)
            pred = logits.max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred+1).astype(np.uint8)
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                logits_copy = logits.squeeze().permute(1,2,0)
                np.savez(os.path.join(opts.save_val_results_to,'deeplab_logits', img_name.split('.')[0]+'.npz'), logits_copy.cpu().numpy())

                cv2.imwrite(os.path.join(opts.save_val_results_to,'deeplab', img_name), (pred+1).astype(np.uint8))
                colorized_preds.save(os.path.join(opts.save_val_results_to,'deeplab_vis', img_name))

if __name__ == '__main__':
    setup_seed(42)
    lis_name_scenes = ['scene0025_00', 'scene0426_00', 'scene0580_00', \
                   'scene0015_00', 'scene0169_00', 'scene0414_00']
    for scene in lis_name_scenes:
        print(f'**process scene: {scene}**\n')
        for data_mode in ['train', 'test']:
            print(f'predict semantic: {data_mode}')
            main(scene, data_mode)
