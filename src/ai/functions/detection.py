import argparse
from tqdm.auto import tqdm

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

os.environ['TRAIN_DIR'] = os.path.join(os.path.dirname(__file__), '../data/train/')
os.environ['VAL_DIR'] = os.path.join(os.path.dirname(__file__), '../data/val/')
os.environ['TEST_DIR'] = os.path.join(os.path.dirname(__file__), '../data/test/')

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

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)

            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:  # if tbox has no matching pbox (FN)
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))

                    # get tbox_pbox logs
                    if save_ans_log:
                        gn = np.tile(shapes[si][0][::-1], 2)
                        tbox_original = xywh2xyxy(labels.clone()[:, 1:5]) * whwh
                        tbox_original = scale_coords(img[si].shape[1:], tbox_original, shapes[si][0], shapes[si][1])
                        tbox_original = (xyxy2xywh(tbox_original) / gn).tolist()
                        tbox_class = tcls
                        
                        print("\ntbox_original : ",tbox_original,"\n")
                        print("\ntbox_class",tbox_class,"\n")

                        # write log by imagefile - target box
                        for k in range(0, nl):
                            row = write_log(paths[0], tbox_class[k], tbox_original[k], np.nan, np.nan, np.nan, const_FN)

                            list_row.append(pd.DataFrame([row]))

                continue

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    if conf >= threshod :
                        tRet_lst.append(int(cls.item()))
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # pbox matched tbox index
            list_pbox_matched_tbox = [-1 for i in range(len(pred))]

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d.item())

                                list_pbox_matched_tbox[pi[j]] = d

                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # get tbox_pbox logs
            if save_ans_log:
                gn = np.tile(shapes[si][0][::-1], 2)
                tbox_original = tbox.clone()[:, :4]
                tbox_original = scale_coords(img[si].shape[1:], tbox_original, shapes[si][0], shapes[si][1])
                tbox_original = (xyxy2xywh(tbox_original) / gn).tolist()
                tbox_class = tcls

                pbox_original = pred.clone()
                pbox_original = scale_coords(img[si].shape[1:], pbox_original[:, :4], shapes[si][0], shapes[si][1])
                pbox_original = (xyxy2xywh(pbox_original) / gn).tolist()
                pbox_class = pred[:, 5].tolist()
                pbox_conf = pred[:, 4].tolist()

                list_correct_iou50 = correct[:, 0].tolist()

                # write log by imagefile - target box
                for k in range(0, nl):

                    # if k in detected:  # if tbox has a matching pbox at IOU : 0.5 (TP)
                    if k in list_pbox_matched_tbox:  # if tbox has a matching pbox at IOU : 0.5 (TP)
                        pbox_index = list_pbox_matched_tbox.index(k)

                        if list_correct_iou50[pbox_index] == True:
                            row = write_log(paths[0], tbox_class[k], tbox_original[k],
                                            pbox_class[pbox_index], pbox_original[pbox_index],
                                            pbox_conf[pbox_index],
                                            const_TP)
                        else:  # if pbox has not enough iou (FP)
                            row = write_log(paths[0], tbox_class[k], tbox_original[k],
                                            pbox_class[pbox_index], pbox_original[pbox_index],
                                            pbox_conf[pbox_index],
                                            const_FP)
                    else:  # if tbox has no matching pbox (FN)
                        row = write_log(paths[0], tbox_class[k], tbox_original[k], np.nan, np.nan, np.nan, const_FN)

                    list_row.append(pd.DataFrame([row]))

                # if pbox has not matching tbox (FP)
                for pbox_index in [i for i, v in enumerate(list_pbox_matched_tbox) if v == -1]:
                    row = write_log(paths[0], np.nan, np.nan,
                                    pbox_class[pbox_index], pbox_original[pbox_index], pbox_conf[pbox_index],
                                    const_FP)
                    list_row.append(pd.DataFrame([row]))



        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
            plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions
            #plot_images(img, output_to_target(output.cpu().numpy(), width, height), paths, f, names)  # predictions


    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # W&B logging
    if plots and wandb:
        wandb.log({"Images": wandb_images})
        wandb.log({"Validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


    # Return results
    if not training:
        print('Results saved to %s' % save_dir)
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    if save_ans_log:
        df = pd.concat(list_row)
        df['target_xywh'] = df.apply(lambda x: np.round(x['target_xywh'], 5), axis=1)
        df['pred_xywh'] = df.apply(lambda x: np.round(x['pred_xywh'], 5), axis=1)
        df['FILENAME'] = df['FILENAME'].str.extract(r'([^\\]+$)')

        n_target_by_cls = df['target_class'].value_counts().sort_index()

        list_df = []
        list_ap = []
        for i, cls in enumerate(n_target_by_cls.index):
            df_tmp = df.loc[df['pred_class'] == cls]
            df_tmp = df_tmp.sort_values('confidence', ascending=False)

            n_tp, n_fp, c_p, c_r = get_p_r_data(df_tmp, cls, n_target_by_cls[cls])

            df_tmp['n_tp'] = n_tp
            df_tmp['n_fp'] = n_fp
            df_tmp['c_p'] = c_p
            df_tmp['c_r'] = c_r

            # calculate ap50
            ap, mpre, mrec = compute_ap(c_r, c_p)
            print('class :  {0} > target : {1}  /  AP@0.5 : {2:.5f}'.format(i, n_target_by_cls[cls], ap))
            list_ap.append(ap)
            list_df.append(df_tmp)

            # plot p-r curve
            px, py = np.linspace(0, 1, 1000), []  # for plotting
            py.append(np.interp(px, mrec, mpre))

            fname = 'precision-recall_curve_class_{0}.png'.format(i)

            # py = np.stack(py, axis=1)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(px, py[0], linewidth=0.5, color='grey')  # plot(recall, precision)
            ax.plot(px, py[0], linewidth=2, color='blue', label='class {0} AP@0.5 {1:.4f}'.format(i, ap))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.legend()
            fig.tight_layout()
            fig.savefig(save_dir / fname, dpi=200)

        print('class :  all > target : {0}  /  AP@0.5 : {1:.5f}'.format(np.sum(n_target_by_cls), np.mean(list_ap)))
        df_export = pd.concat(list_df)
        df_export.to_csv(save_dir / 'test_log.csv', index=False)


    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    return tRet_lst
