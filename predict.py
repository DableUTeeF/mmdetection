from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import copy
import os.path as osp
from mmdet.apis import set_random_seed

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


@DATASETS.register_module()
class AlgeaDataset(CustomDataset):
    CLASSES = ('mif', 'ov')

if __name__ == '__main__':
    dataset = AlgeaDataset('anns/train.json', [])
    x = dataset[0]

    cfg = Config.fromfile('./configs/detr/detr_r50_8x4_150e_coco.py')

    # Modify dataset type and path
    cfg.dataset_type = 'AlgeaDataset'
    cfg.data_root = ''

    cfg.data.test.type = 'AlgeaDataset'
    cfg.data.test.data_root = ''
    cfg.data.test.ann_file = '/home/palm/PycharmProjects/mmdetection/anns/test.json'
    cfg.data.test.img_prefix = ''

    cfg.data.train.type = 'AlgeaDataset'
    cfg.data.train.data_root = ''
    cfg.data.train.ann_file = '/home/palm/PycharmProjects/mmdetection/anns/train.json'
    cfg.data.train.img_prefix = ''

    cfg.data.val.type = 'AlgeaDataset'
    cfg.data.val.data_root = ''
    cfg.data.val.ann_file = '/home/palm/PycharmProjects/mmdetection/anns/test.json'
    cfg.data.val.img_prefix = ''

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)


    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = init_detector(cfg, '/home/palm/PycharmProjects/mmdetection/tutorial_exps/epoch_24.pth', device='cpu')
    # test a single image
    result = inference_detector(model, '/media/palm/data/MicroAlgae/22_11_2020/images/00001.jpg')
    # show the results
    show_result_pyplot(model, '/media/palm/data/MicroAlgae/22_11_2020/images/00001.jpg', result, score_thr=0.3)


