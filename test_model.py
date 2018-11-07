import postprocess
import cv2
import numpy as np
import datasets
import torch
from torch.utils.data import DataLoader
import net
import torch.nn as nn
import ImgLib.ImgFormat as ImgFormat
import ImgLib.ImgTransform as ImgTransform
import os


def cal_label_on_batch(my_net, imgs, version="2s"):
    scale = 2 if version == "2s" else 4
    with torch.no_grad():
        out_1, out_2 = my_net.forward(imgs)
    all_boxes = postprocess.mask_to_box(out_1, out_2, scale=scale)
    return all_boxes


def cal_IOU(box1, box2):
    """
    box1, box2: list or numpy array of size 4*2 or 8, h_index first
    """
    box1 = np.array(box1).reshape([1, 4, 2])
    box2 = np.array(box2).reshape([1, 4, 2])
    box1_max = box1.max(axis=1)
    box2_max = box2.max(axis=1)
    w_max = max(box1_max[0][0], box2_max[0][0])
    h_max = max(box1_max[0][1], box2_max[0][1])
    canvas = np.zeros((h_max + 1, w_max + 1))
    # print(canvas.shape)
    box1_canvas = canvas.copy()
    box1_area = np.sum(cv2.drawContours(box1_canvas, box1, -1, 1, thickness=-1))
    # print(box1_area)
    box2_canvas = canvas.copy()
    box2_area = np.sum(cv2.drawContours(box2_canvas, box2, -1, 1, thickness=-1))
    # print(box2_area)
    cv2.drawContours(canvas, box1, -1, 1, thickness=-1)
    cv2.drawContours(canvas, box2, -1, 1, thickness=-1)
    union = np.sum(canvas)
    # print(union)
    intersction = box1_area + box2_area - union
    return intersction / union


def comp_gt_and_output(my_labels, gt_labels, threshold):
    """
    return: [true_pos, false_pos, false_neg]
    """
    coor = gt_labels["coor"]
    ignore = gt_labels["ignore"]
    true_pos, true_neg, false_pos, false_neg = [0] * 4
    for my_label in my_labels:
        for gt_label in coor:
            if cal_IOU(my_label, gt_label) > threshold:
                true_pos += 1
                break
        else:
            false_pos += 1
    for i, gt_label in enumerate(coor):
        if ignore[i]:
            continue
        for my_label in my_labels:
            if cal_IOU(gt_label, my_label) > threshold:
                break
        else:
            false_neg += 1
    return true_pos, false_pos, false_neg


def test(my_net, exp_dir, epoch, results_dir,
         images_dir, labels_dir, num_images,
         mean, version, image_size=(512, 512),
         gpu=True, multi_gpu=False, vis_per_img=10):
    dataset = datasets.PixelLinkIC15Dataset(images_dir, labels_dir, train=False,
                                            all_trains=num_images, version=version, mean=mean,
                                            image_size_test=image_size)
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    if gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        if multi_gpu:
            my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(os.path.join(exp_dir, 'snapshots', 'epoch_%08d.mdl' % epoch))
    my_net.load_state_dict(checkpoint['state_dict'])
    my_net.eval()

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    true_pos, true_neg, false_pos, false_neg = [0] * 4
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image'].to(device)
        image = image.unsqueeze(0)
        my_labels = cal_label_on_batch(my_net, image)[0]
        # print("my labels num: %d" % len(my_labels))
        res = comp_gt_and_output(my_labels, sample["label"], 0.5)
        if i % vis_per_img == 0:
            image = image.squeeze(0).cpu().numpy()
            image = ImgFormat.ImgOrderFormat(image, from_order="CHW", to_order="HWC")
            image = ImgTransform.UnzeroMeanImage(image, mean[0], mean[1], mean[2])
            image = ImgFormat.ImgColorFormat(image, from_color="RGB", to_color="BGR")
            # color : gt = red, ignore = yellow, detection = blue
            image = visualize_label(image, sample["label"]["coor"], color=(0, 0, 255), ignore=sample["label"]["ignore"])
            image = visualize_label(image, my_labels, color=(255, 0, 0), thickness=2)
            cv2.imwrite("%s/img_%d.jpg" % (results_dir, i), image)
        true_pos += res[0]
        false_pos += res[1]
        false_neg += res[2]
        if (true_pos + false_pos) > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0
        if (true_pos + false_neg) > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0
        print("i: %d, TP: %d, FP: %d, FN: %d, precision: %f, recall: %f" % (i, true_pos, false_pos, false_neg, precision, recall))

    perf_str = "TP: %d, FP: %d, FN: %d, precision: %f, recall: %f (P=%d)" % (
        true_pos, false_pos, false_neg, precision, recall, (true_pos + false_neg))
    os.system('echo "%s" > %s' % (perf_str, os.path.join(results_dir, 'performance.txt')))

    perf_str2 = "%d, %d,%d,%d,%f,%f" % (epoch, true_pos, false_pos, false_neg, precision, recall)
    test_file = os.path.join(exp_dir, 'performance-%s.csv' % images_dir.split('/')[0])
    if not os.path.exists(test_file):
        os.system('echo "epoch,TP,FP,FN,precision,recall" > %s' % test_file)
    os.system('echo "%s" >> %s' % (perf_str2, test_file))


def visualize_label(img, boxes, color=(0, 255, 0), ignore=None, thickness=1):
    """
    img: HWC
    boxes: array of num * 4 * 2
    """
    boxes = np.array(boxes).reshape(-1, 4, 2)
    img = np.ascontiguousarray(img)
    if ignore is None:
        cv2.drawContours(img, boxes, -1, color, thickness=thickness)
    else:
        for box, is_ignore in zip(boxes, ignore):
            color_box = (0, 255, 255) if is_ignore else color
            cv2.drawContours(img, np.expand_dims(box, 0), -1, color_box, thickness=thickness)
    return img
