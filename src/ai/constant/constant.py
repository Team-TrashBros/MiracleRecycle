import os
import sys

os.environ['TRAIN_DIR'] = os.path.join(os.path.dirname(__file__), '../data/train/')
os.environ['VAL_DIR'] = os.path.join(os.path.dirname(__file__), '../data/val/')
os.environ['TEST_DIR'] = os.path.join(os.path.dirname(__file__), '../data/test/')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from tqdm.auto import tqdm

from utils.datasets import create_dataloader
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
torch.manual_seed(777)

threshod = 0.5

opt = argparse.Namespace(
    # weights = ['runs/train/newconn_model/weights/last.pt'],
    weights= [os.path.join(os.path.dirname(__file__),'../runs/train/newconn_model/weights/last.pt')],
    data = os.path.join(os.path.dirname(__file__),'../data/newconn.yaml'),
    batch_size = 16,
    img_size = 640,
    conf_thres=0.01,
    iou_thres=0.5,  # for NMS
    task='test',
    device = 'cpu',
    single_cls=False,
    augment=False,
    verbose=True,
    save_txt=True,
    save_conf=True,
    save_json=False,
    save_ans_log=False,
    project= os.path.join(os.path.dirname(__file__),'../runs/test'),
    name='newconn_test',
    exist_ok=False,
    # cfg='src/ai/cfg/yolor_p6.cfg',
    cfg= os.path.join(os.path.dirname(__file__),'../cfg/yolor_p6.cfg'),
    names= os.path.join(os.path.dirname(__file__),'../data/newconn.names')
)

data = opt.data
weights = opt.weights
if opt.save_ans_log:
    batch_size = 1
else:
    batch_size = opt.batch_size
imgsz = opt.img_size
conf_thres = opt.conf_thres
iou_thres = opt.iou_thres
save_json = opt.save_json
save_ans_log = opt.save_ans_log
single_cls = opt.single_cls
augment = opt.augment
verbose = opt.verbose
model = None
dataloader = None
save_dir = Path(''),  # for saving images
save_txt = opt.save_txt,  # for auto-labelling
save_conf = opt.save_conf,
plots = True,
log_imgs = 0  # number of logged images
