import net
import torch
import torch.nn as nn
import datasets
import math
from torch import optim
from criterion import PixelLinkLoss
import postprocess
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
import os
import sys
import time
import argparse
import importlib
import shutil
import glob
from test_model import test

sys.path.append('torchsample')
from torchsample import callbacks
from torchsample.modules import ModuleTrainer

parser = argparse.ArgumentParser(description='')
parser.add_argument('exp', type=str, help='experiment name')
parser.add_argument('--mode', type=str, default='train', help='train / retrain / benchmark-train / benchamrk-test')
parser.add_argument('--epoch', type=int, default=None, help='epoch to be chosen (relevant only for benchmark modes)')
# parser.add_argument('change', metavar='N', type=int, help='an integer for change')
args = parser.parse_args()

exp_name = args.exp
config = importlib.import_module('configs.%s' % exp_name)
if not hasattr(config, 'net_params'):
    config.net_params = {'version': config.version}
config.net_params['version'] = config.version
out_dir = "results/%s" % exp_name


def train(epoch, dataloader, my_net, loss, optimizer, scheduler, device, start_epoch=0,
          test_datasets={}, vis_per_img=10):
    global trainer, callbacks_cont

    logs = {'batch_size': config.batch_size, 'num_batches': len(dataloader),
            'num_epoch': config.epoch,
            'has_val_data': False, 'has_regularizers': False}
    callbacks_cont.on_train_begin(logs)

    iteration = 0
    for i_epoch in range(start_epoch, epoch):
        epoch_logs = {}
        callbacks_cont.on_epoch_begin(i_epoch, epoch_logs)
        for i_batch, sample in enumerate(dataloader):
            callbacks_cont.on_batch_begin(i_batch)
            scheduler.step(epoch=i_epoch)
            start = time.time()
            images = sample['image'].to(device)
            # print(images.shape, end=" ")
            pixel_masks = sample['pixel_mask'].to(device)
            neg_pixel_masks = sample['neg_pixel_mask'].to(device)
            link_masks = sample['link_mask'].to(device)
            pixel_pos_weights = sample['pixel_pos_weight'].to(device)

            out_cls, out_link = my_net.forward(images)
            # print(out_2)

            total_loss, pixel_loss, link_loss = loss(out_cls, out_link, pixel_masks, link_masks, neg_pixel_masks, pixel_pos_weights)
            #print("iteration %d : " % iteration)  #, end=": ")
            #print("pixel_loss: " + str(pixel_loss.tolist()))  #, end=", ")
            # print("pixel_loss_pos: " + str(pixel_loss_pos.tolist()), end=", ")
            # print("pixel_loss_neg: " + str(pixel_loss_neg.tolist()), end=", ")
            #print("link_loss: " + str(link_loss.tolist()))  #, end=", ")
            # print("link_loss_pos: " + str(link_loss_pos.tolist()), end=", ")
            # print("link_loss_neg: " + str(link_loss_neg.tolist()), end=", ")
            #print("total loss: " + str(total_loss.tolist()))  #, end=", ")

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            end = time.time()
            # print("time: " + str(end - start))
            iteration += 1

            batch_logs = {'loss': total_loss.tolist()}
            callbacks_cont.on_batch_end(i_batch, batch_logs)

        if i_epoch > 0 and i_epoch % 50 == 0:
            checkpoint = {'epoch': i_epoch,
                          'state_dict': my_net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(out_dir, 'snapshots', 'epoch_%08d.mdl' % i_epoch))

            for benchmark_name, datasets_test in test_datasets.items():
                results_dir = os.path.join(out_dir, '%s_e%08d' % (benchmark_name, i_epoch))
                test_file = os.path.join(out_dir, 'performance-%s.csv' % datasets_test.images_dir.split('/')[0])
                test(my_net, datasets_test, i_epoch, out_dir, results_dir, test_file,
                     gpu=config.gpu, multi_gpu=config.multi_gpu, vis_per_img=vis_per_img,
                     weights_preloaded=True)
                my_net.train()

        epoch_logs.update(trainer.history.batch_metrics)
        callbacks_cont.on_epoch_end(i_epoch, logs=epoch_logs)


def main(retrain=False, test_datasets={}, vis_per_img=10):
    res_dir = os.path.join(out_dir, 'snapshots')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # shutil.copyfile(os.path.join('configs', '%s.py' % exp_name), os.path.join(out_dir, 'config.py'))
    with open(os.path.join(out_dir, 'config.py'), 'w') as f:
        params_names = config.__dir__()
        for param_name in params_names:
            if param_name.startswith('__'):
                continue
            param_value = getattr(config, param_name)
            if isinstance(param_value, str):
                f.write('%s = "%s"\n' % (param_name, param_value))
            else:
                f.write('%s = %s\n' % (param_name, param_value))

    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir,
                                            all_trains=config.all_trains, version=config.version,
                                            mean=config.mean, use_rotate=config.use_rotate, use_crop=config.use_crop,
                                            image_size_train=config.image_size_train,
                                            image_size_test=config.image_size_test)
    # sampler = WeightedRandomSampler([1/len(dataset)]*len(dataset), config.batch_size, replacement=True)
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=6)
    model = net.PixelLinkNet(**config.net_params)  #net.Net(config.version, config.dilation)

    if config.gpu:
        device = torch.device("cuda:0")
        model = model.cuda()
        if config.multi_gpu:
            model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")

    loss = PixelLinkLoss(config.pixel_weight, config.link_weight, config.neg_pos_ratio)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate1, momentum=config.momentum, weight_decay=config.weight_decay)
    epoch_milestone = math.ceil(config.step2_start / len(dataloader))
    print('LR schedule')
    print('[%05d - %05d] : %E' % (0, epoch_milestone, config.learning_rate1))
    print('[%05d - %05d] : %E' % (epoch_milestone, config.epoch, config.learning_rate2))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [epoch_milestone], config.learning_rate2 / config.learning_rate1)

    global trainer, callbacks_cont
    tqdm = callbacks.TQDM()
    log_path = os.path.join(out_dir, 'log_train.csv')
    index = 0
    while os.path.exists(log_path):
        index += 1
        log_path = os.path.join(out_dir, 'log_train_%02d.csv' % index)

    logger = callbacks.CSVLogger(log_path)
    trainer = ModuleTrainer(model)
    trainer.compile(optimizer, loss, callbacks=[tqdm, logger])
    callbacks_cont = callbacks.CallbackContainer(trainer._callbacks)
    callbacks_cont.set_trainer(trainer)

    if retrain:
        # find latest snapshot
        snapshots_dir = os.path.join(out_dir, 'snapshots')
        model_files = glob.glob(snapshots_dir + '/epoch_*')
        if model_files:
            resume_path = sorted(model_files)[-1]
            start_epoch = 1 + int(os.path.basename(resume_path)[len('epoch_'):-4])
            print('Loading snapshot from : %s' % resume_path)
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            # couldnt find snapshots
            start_epoch = 0
    else:
        start_epoch = 0

    train(config.epoch, dataloader, model, loss, optimizer, scheduler, device, start_epoch=start_epoch,
          test_datasets=test_datasets, vis_per_img=vis_per_img)


if __name__ == "__main__":
    if args.mode in ['train', 'retrain']:
        dataset_train = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir,
                                                      train=False, all_trains=config.all_trains,
                                                      version=config.version, mean=config.mean,
                                                      image_size_test=config.image_size_test)
        vis_per_img_train = int(math.ceil(config.all_trains / 100.0))

        dataset_test = datasets.PixelLinkIC15Dataset(config.test_images_dir, config.test_labels_dir,
                                                      train=False, all_trains=config.all_tests,
                                                      version=config.version, mean=config.mean,
                                                      image_size_test=config.image_size_test)
        vis_per_img_test = int(math.ceil(config.all_tests / 100.0))

        retrain = args.mode == 'retrain'

        main(retrain=retrain,
             test_datasets={'benchmark-train': dataset_train, 'benchmark-test': dataset_test},
             vis_per_img=vis_per_img_train)
    else:
        epoch = args.epoch  #config.test_model_index
        assert epoch is not None
        model = net.PixelLinkNet(**config.net_params)

        if args.mode == 'benchmark-test':
            images_dir = config.test_images_dir
            labels_dir = config.test_labels_dir
            num_images = config.all_tests
            results_dir = os.path.join(out_dir, 'benchmark-test_e%08d' % epoch)
        elif args.mode == 'benchmark-train':
            images_dir = config.train_images_dir
            labels_dir = config.train_labels_dir
            num_images = config.all_trains
            results_dir = os.path.join(out_dir, 'benchmark-train_e%08d' % epoch)

        vis_per_img = int(math.ceil(num_images / 100.0))

        test_file = os.path.join(out_dir, 'performance-%s.csv' % images_dir.split('/')[0])
        dataset = datasets.PixelLinkIC15Dataset(images_dir, labels_dir, train=False,
                                                all_trains=num_images, version=config.version,
                                                mean=config.mean,
                                                image_size_test=config.image_size_test)
        test(model, dataset, epoch, out_dir, results_dir, test_file,
             gpu=config.gpu, multi_gpu=config.multi_gpu, vis_per_img=vis_per_img)
        # test_model()
