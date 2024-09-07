import argparse
from tqdm.auto import tqdm

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.datasets import create_dataloader
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from constant.constant import *

import os
import sys

from models.models import *

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
torch.manual_seed(777)

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

# def write_log(filename, target_class, target_coord, pred_class, pred_coord, conf, ans_iou, c_TP, c_FP, c_P, c_R):
def write_log(filename, target_class, target_coord, pred_class, pred_coord, conf, ans_iou):
    row = {'FILENAME': filename,
           'target_class': target_class,
           'target_xywh': target_coord,
           'pred_class': pred_class,
           'pred_xywh': pred_coord,
           'confidence': conf,
           'ans_iou': dict_answer[ans_iou],
           # 'c_TP' : c_TP,
           # 'c_FP' : c_FP,
           # 'c_P' : c_P,
           # 'c_R' : c_R
           }
    return row


def get_p_r_data(df, cls, n_target):
    df = df.loc[df['pred_class'] == cls]
    n_tp = np.repeat(0, len(df))
    n_fp = np.repeat(0, len(df))
    c_p = np.repeat(0, len(df))
    c_r = np.repeat(0, len(df))

    i = 0
    for index, row in df.iterrows():
        if row['ans_iou'] == 'TP':
            n_tp[i:] += 1
        if row['ans_iou'] == 'FP':
            n_fp[i:] += 1
        # if not np.isnan(row['target_class']):
        #   n_target[i:] += 1
        i += 1

    c_p = n_tp / (n_tp + n_fp)
    c_r = n_tp / n_target

    return n_tp, n_fp, c_p, c_r

dict_answer = {0: 'TP', 1: 'FN', 2: 'FP'}  # 'TN' does not exists for object detection.
const_TP = 0
const_FN = 1
const_FP = 2
