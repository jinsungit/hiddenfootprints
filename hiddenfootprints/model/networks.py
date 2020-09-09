"""HiddenFootprints walkability prediction network module

Based on a Resnet + UNet structure to predict where people can walk in a scene.
"""

import torch
import torch.nn as nn
from .resnet import ResUNet
import numpy as np
import torchvision.transforms as transforms
import cv2


class GeneratorHeatMap(nn.Module):
    """Network model class
    
    Based on ResUNet.
    """
    def __init__(self):
        super(GeneratorHeatMap, self).__init__()
        num_in_layers = 3

        self.backbone = ResUNet(encoder='resnet50', pretrained=True, num_in_layers=num_in_layers, num_out_layers=1)
        self.backbone.cuda()

    def forward(self, img, return_ft=False):
        # if return_ft==True, out[0] is prediction, out[1] is feature map 
        out = self.backbone(img, return_ft=return_ft)
        return out

class FootprintsPredictor(nn.Module):
    """A wrapper class to load model, test on single image.
    
    """
    def __init__(self):
        super(FootprintsPredictor, self).__init__()
        # model
        self.netG = GeneratorHeatMap()

    def load_model(self, model_file):
        self.netG.load_state_dict(torch.load(model_file))
        print('Loaded model: {}.'.format(model_file))

    # given an image, return output
    def forward(self, img, return_ft=False):
        out = self.netG(img, return_ft=return_ft)
        pred = {}
        if return_ft:
            pred['locmap'], pred['ftmaps'] = out
        else:
            pred['locmap'] = out
            pred['ftmaps'] = None
        return pred


    def test_single_im(self, img):
        """Test the model on a single image. img size: hxwx3

        """       
        # data transform
        data_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225))])

        h,w = img.shape[:2]
        resize_factor = 480. / h

        img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        w_new = img.shape[1]

        # crops and their weights
        center_crop_left = (w_new-640)//2
        center_crop_right = (w_new+640)//2
        center_center = w_new//2

        left_weights = np.zeros((1, w_new//4))
        left_weights[0, :center_crop_left//4] = 1.0
        left_weights[0, center_crop_left//4:center_center//4] = np.linspace(1, 0, center_center//4 - center_crop_left//4)
        left_weights_map = np.tile(left_weights, (480//4, 1))

        center_weights = np.zeros((1, w_new//4))
        center_weights[0, center_crop_left//4:center_center//4] = np.linspace(0, 1, center_center//4 - center_crop_left//4)
        center_weights[0, center_center//4:center_crop_right//4] = np.linspace(1, 0, center_crop_right//4 - center_center//4)
        center_weights_map = np.tile(center_weights, (480//4, 1))

        right_weights = np.zeros((1, w_new//4))
        right_weights[0, center_crop_right//4:] = 1.0
        right_weights[0, center_center//4:center_crop_right//4] = np.linspace(0, 1, center_crop_right//4 - center_center//4)
        right_weights_map = np.tile(right_weights, (480//4, 1))

        weights_map = [left_weights_map, center_weights_map, right_weights_map]

        # take weighted three fix-sized crops
        x_crops = [0, (w_new-640)//2, w_new-640]
        pred_map_whole = np.zeros((480//4,w_new//4))
        for (x_crop_i,x_crop) in enumerate(x_crops):
            img_cropped = img[:, x_crop:x_crop+640]
            pred_map_cur = np.zeros(pred_map_whole.shape)

            # convert to tensor
            img_cropped = data_transform(img_cropped).float()
            real_img = img_cropped.unsqueeze(0).cuda()

            pred_map = self(real_img)['locmap'].squeeze().detach().cpu().numpy() # hxw

            # merge
            x_crop = int(x_crop//4)
            pred_map_cur[:, x_crop:x_crop+640//4] = pred_map
            pred_map_whole += pred_map_cur * weights_map[x_crop_i]

        # resize it back to image size
        pred_map_whole = cv2.resize(pred_map_whole, None, fx=4/resize_factor, fy=4/resize_factor, interpolation=cv2.INTER_AREA)

        return pred_map_whole


