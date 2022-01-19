# Copyright (c) OpenMMLab. All rights reserved.
import torch
import mmcv
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['masked_img'], str):
            results['filename'] = results['masked_img']
            results['ori_filename'] = results['masked_img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['masked_img'])
        results['masked_img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class LoadMask:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        mask = mmcv.imread(results['mask'])
        results['mask'] = mask
        return results

def inpainting_inference2(model, masked_img, mask):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        masked_img (str): File path of image with mask.
        mask (str): Mask file path.

    Returns:
        Tensor: The predicted inpainting result.
    """

    infer_pipeline = [
        dict(type='Pad', keys=['masked_img', 'mask'], mode='reflect'),
        dict(
            type='Normalize',
            keys=['masked_img'],
            mean=[127.5] * 3,
            std=[127.5] * 3,
            to_rgb=False),
        dict(type='GetMaskedImage', img_name='masked_img'),
        dict(
            type='Collect',
            keys=['masked_img', 'mask'],
            meta_keys=['masked_img']),
        dict(type='ImageToTensor', keys=['masked_img', 'mask'])
    ]

    # build the data pipeline
    infer_pipeline = [LoadImage(), LoadMask()] + infer_pipeline
    test_pipeline = Compose(infer_pipeline)
    # prepare data
    data = dict(masked_img=masked_img, mask=mask)
    data = test_pipeline(data)

    # data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        device = next(model.parameters()).device  # model device
        data = scatter(data, [device])[0]
    # else:
    #     data = scatter(data, [-1])[0]

    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['fake_img']
