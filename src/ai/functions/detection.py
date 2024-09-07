import argparse
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from utils.datasets import create_dataloader
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from constant.constant import *
from functions.Basefunctions_for_ai import *

def detection(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         save_ans_log=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0):  # number of logged images

    tRet_lst = []
    
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(Path(opt.project) / opt.name)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = Darknet(opt.cfg).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=True)[0]

    seen = 0
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # 새로 추가된 초기값
    list_row = []
    
    class_counter = {}
    
    os.path.join(os.path.dirname(__file__), '..', 'test', 'result')
    
    result_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'test', 'result'))
    result_dir.mkdir(parents=True, exist_ok=True)
    
    if result_dir.exists() and result_dir.is_dir():
        # Delete all files in the result directory
        for file in result_dir.iterdir():
            if file.is_file():
                file.unlink()  # Remove the file

    # Recreate the result directory if it was deleted
    result_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # Normalize image
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        with torch.no_grad():
            # Run model and get output
            inf_out, train_out = model(img, augment=augment)
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)

        # Process each image in the batch
        for si, pred in enumerate(output):
            path = Path(paths[si])  # current image path
            if len(pred) == 0:
                continue

            # Convert to NumPy format and ensure proper scaling and type
            img0 = (img[si].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Iterate over predictions for each image
            for i, det in enumerate(pred):
                # Bounding box coordinates, confidence score, and class index
                xyxy = det[:4].cpu().numpy().astype(int)
                conf = det[4].cpu().item()
                cls = int(det[5].cpu().item())

                if conf >= threshod :
                    tRet_lst.append(cls)
                    # Unpack the bounding box
                    x1, y1, x2, y2 = xyxy
                    print("Original x1, y1, x2, y2:", x1, y1, x2, y2)

                    # Clamp coordinates to image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img0.shape[1], x2)
                    y2 = min(img0.shape[0], y2)

                    # Check if the bounding box is valid
                    if x1 >= x2 or y1 >= y2:
                        print(f"Skipping invalid bounding box: {x1}, {y1}, {x2}, {y2}")
                        continue

                    # Crop the detected object
                    cropped_img = img0[y1:y2, x1:x2, :]
                    print(f"Cropped image shape: {cropped_img.shape}, dtype: {cropped_img.dtype}")

                    # Ensure valid cropping
                    if cropped_img.size == 0:
                        print("Invalid crop, skipping.")
                        continue

                    # Convert NumPy array to PIL image
                    pil_img = Image.fromarray(cropped_img)

                    # Get the class name (use the index, or modify this to map to actual class names)
                    class_name = f'{cls}'

                    # Initialize class counter if not already initialized
                    if class_name not in class_counter:
                        class_counter[class_name] = 0

                    # Increment the class counter
                    class_counter[class_name] += 1

                    # Create the filename: class_name + "__" + order
                    crop_filename = f"{class_name}__{class_counter[class_name]}.jpg"
                    crop_path = result_dir / crop_filename
                    print(f"Attempting to save cropped image to: {crop_path}")

                    # Save the cropped image using PIL
                    pil_img.save(str(crop_path))

                    print(f"Cropped image saved at: {crop_path}")

                    # Optionally display the cropped image for verification
                    # pil_img.show()

    return tRet_lst
