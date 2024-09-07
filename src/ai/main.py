import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Basefunctions_for_ai import *
from functions.detection import *

opt.data = check_file(opt.data)  # check file
print(opt)

if opt.task in ['val', 'test']:  # run normally

    now = datetime.now()
    print("Model Test Start at ", now)
    print('\n')

    detection(opt.data,
         opt.weights,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt.save_json,
         opt.save_ans_log,
         opt.single_cls,
         opt.augment,
         opt.verbose,
         save_txt=opt.save_txt,
         save_conf=opt.save_conf,
         )

    now = datetime.now()
    print('\n')
    print("Model Test End at ", now)