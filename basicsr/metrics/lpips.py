import cv2
import numpy as np
import torch

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


import lpips

loss_fn_alex = lpips.LPIPS(net='alex', version='0.1')


# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def im2tensor(image, imtype=np.uint8, cent=1., factor=255. / 2.):

    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False):

    assert img.shape == img2.shape, (f'Image shapes are differnet: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)


    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = im2tensor(img[:,:,::-1])
    img2 = im2tensor(img2[:,:,::-1])

    d = loss_fn_alex(img, img2)

    if d == 0:
        return float('inf')
    return float(d)
