import os
import time
import logging
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from torchvision.models import resnet18 as imagenet_resnet18
from torchvision.models import resnet50 as imagenet_resnet50
from torchvision.models import resnet101 as imagenet_resnet101

from metrics.metrics import grad_norm, eigen_spec
from models import cifar_resnet50, cifar_resnet18, cifar_resnet101, cifar_wrn28_10

from util.sam import SAM
from util.dataset import CIFAR
from metrics.accuracy import accuracy
from util.logger import CSVLogger, AverageMeter, get_device
from util.utils import log_metric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset')
parser.add_argument('--model', default='resnet18')
parser.add_argument("--aug", default='basic', type=str, choices=['basic', 'cutout', 'autoaugment'],help='Data augmentation')
parser.add_argument('--epochs', type=int, default=200, help='Epochs')
parser.add_argument('--alpha', type=float, default=1e5, help='alpha parameter for regularization')
parser.add_argument('--rho', type=float, default=0.05, help='rho parameter for SAM')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--mo', type=float, default=0, help='momentum')
parser.add_argument('--wd', type=float, default=0, help='weight decay')
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--loadckpt', default=False, action='store_true')

parser.add_argument('--project_name', default='eigen test', action='store_true')
parser.add_argument('--exp_name', default='sam', action='store_true')
parser.add_argument('--num_limit', type=int, default=200, help='max number for train per round')
args = parser.parse_args()

if args.dataset == 'cifar10':
    args.num_classes = 10
    args.milestones = [100, 120]
    args.data_dir = f"./data/{args.dataset}"
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.milestones = [100, 150]
    args.data_dir = f"./data/{args.dataset}"
elif args.dataset == 'imagenet':
    args.num_classes = 1000
    args.milestones = [30, 60, 90]
    args.data_dir = f"./data/{args.dataset}"
elif args.dataset == 'tinyimagenet':
    args.num_classes = 200
    args.milestones = [30, 60, 90]
    args.data_dir = f"./data/{args.dataset}"
else:
    print(f"BAD COMMAND dtype: {args.dataset}")

#random seed
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Intialize directory and create path
args.ckpt_dir = "../util/"
os.makedirs(args.ckpt_dir, exist_ok=True)
logger_name = os.path.join(args.ckpt_dir, f"gcsam_{args.model}_{args.dataset}_{args.aug}_run{args.seed}")

# Logging tools
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(logger_name + ".log"),
        logging.StreamHandler(),
    ],
)
logging.info(args)

device = get_device()
global_step = 0
def run_one_epoch(phase, loader, model, criterion, optimizer, args):
    loss, acc = AverageMeter(), AverageMeter()
    # h_f_product_p, h_f_product_h, h_f_product_f = AverageMeter(), AverageMeter(), AverageMeter()
    # h_norm, f_norm, p_norm, h_f_norm = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    t = time.time()
    global global_step
    # h_norm, f_norm, p_norm, h_f_norm = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    # h_f_product_p, h_f_product_h, h_f_product_f = AverageMeter(), AverageMeter(), AverageMeter()
    total_num = 0
    for batch_idx, inp_data in enumerate(loader, 1):
        # if total_num >= args.num_limit: break
        inputs, targets = inp_data
        inputs, targets = inputs.to(device), targets.to(device)
        data_size = inputs.size(0)
        if phase == 'train':
            model.train()
            with torch.set_grad_enabled(True):
                # compute output
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                # compute gradient and do SGD step
                optimizer.zero_grad()
                batch_loss.backward(retain_graph=True)

                grad_f = optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets) 

                batch_loss.backward()
                grad_p = optimizer.second_step(zero_grad=True)

                # grad_h = grad_p - grad_f
                # grad_h_f = grad_h - grad_f

                # # 计算内积
                # h_f_product_p = torch.dot(grad_h_f, grad_p).item()
                # h_f_product_h = torch.dot(grad_h_f, grad_h).item()
                # h_f_product_f = torch.dot(grad_h_f, grad_f).item()
                # # 计算范数
                # h_norm = torch.norm(grad_h).item()
                # f_norm = torch.norm(grad_f).item()
                # p_norm = torch.norm(grad_p).item()
                # h_f_norm = torch.norm(grad_h_f).item()
                # if f_norm > h_norm:
                #     h_f_norm_std = h_f_norm / h_norm
                # else:
                #     h_f_norm_std = h_f_norm / f_norm
                # log_metric(
                #     ['h_f_product_p', 'h_f_product_h', 'h_f_product_f',
                #      'h_norm', 'f_norm', 'p_norm', 'h_f_norm', 'h_f_norm_std'],
                #     [h_f_product_p, h_f_product_h, h_f_product_f,
                #         h_norm, f_norm, p_norm, h_f_norm, h_f_norm_std],
                #     global_step,
                # )
                global_step += 1
                total_num += 1

        elif phase == 'val':
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
        else:
            logging.info('Define correct phase')
            quit()

        loss.update(batch_loss.item(), data_size)

        batch_acc = accuracy(outputs, targets, topk=(1,))[0]
        acc.update(float(batch_acc), data_size)

        if batch_idx % args.print_freq == 0:
            info = f"Phase:{phase} -- Batch_idx:{batch_idx}/{len(loader)}" \
                   f"-- {acc.count / (time.time() - t):.2f} samples/sec" \
                   f"-- Loss:{loss.avg:.2f} -- Acc:{acc.avg:.2f}"
            logging.info(info)

    return loss.avg, acc.avg


def main(args):
    dataset = CIFAR(args)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.model == 'resnet50':
            model = cifar_resnet50(num_classes=args.num_classes)
        elif args.model == 'resnet18':
            model = cifar_resnet18(num_classes=args.num_classes)
        elif args.model == 'resnet101':
            model = cifar_resnet101(num_classes=args.num_classes)
        elif args.model == 'wrn':
            model = cifar_wrn28_10(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    elif 'imagenet' in args.dataset:
        if args.model == 'resnet50':
            model = imagenet_resnet50(num_classes=args.num_classes)
        elif args.model == 'resnet18':
            model = imagenet_resnet18(num_classes=args.num_classes)
        elif args.model == 'resnet101':
            model = imagenet_resnet101(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    else:

        print("define dataset type")

    args_dict = dict(vars(args))
    wandb.init(project=args.project_name, name=args.exp_name, config=args_dict)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(model.parameters(), optim.SGD, rho=args.rho, lr=args.lr, momentum=args.mo, weight_decay=args.wd)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    base_optimizer = optimizer.base_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=args.epochs)
    
    csv_logger = CSVLogger(args, ['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'], logger_name + '.csv')

    if args.loadckpt:
        state = torch.load(f"{args.ckpt_dir}/{logger_name}_best.pth.tar")
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best_acc = state['best_acc']
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 0
        best_acc = -float('inf')

    for epoch in range(start_epoch, args.epochs):
        logging.info('Epoch: [%d | %d]' % (epoch, args.epochs))

        trainloss, trainacc = run_one_epoch('train', dataset.train, model, criterion, optimizer, args)
        log_metric(
            ['Train Loss', 'Train Accuracy'],
            [trainloss, trainacc],
            epoch,
            True
        )
        # log_metric(
        #     ['h_f_product_p', 'h_f_product_h', 'h_f_product_f',
        #      'h_norm', 'f_norm', 'p_norm', 'h_f_norm'],
        #     [metrics[0], metrics[1], metrics[2],
        #         metrics[3], metrics[4], metrics[5], metrics[6]],
        #     epoch,
        #     True
        # )
        valloss, valacc = run_one_epoch('val', dataset.test, model, criterion, optimizer, args)
        log_metric(
            ['Test Loss', 'Test Accuracy'],
            [valloss, valacc],
            epoch,
            True
        )
        csv_logger.save_values(epoch, trainloss, trainacc, valloss, valacc)
        if epoch % 10 == 0:
           train_grad_norm = grad_norm(model, criterion, optimizer, dataloader=dataset.train, lp=2)
           train_top_eigen, train_trace = eigen_spec(model, criterion, dataloader=dataset.train)
           test_grad_norm = grad_norm(model, criterion, optimizer, dataloader=dataset.test, lp=2)
           test_top_eigen, test_trace = eigen_spec(model, criterion, dataloader=dataset.test)
           log_metric(
                ['Train Top Eigen', 'Train Trace', 'Train Grad Norm', 'Test Top Eigen', 'Test Trace', 'Test Grad Norm'],
                [train_top_eigen, train_trace, train_grad_norm, test_top_eigen, test_trace, test_grad_norm],
               epoch,
               True
           )

        scheduler.step()

        if valacc > best_acc:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }
            torch.save(state, f"{args.ckpt_dir}/{logger_name}_best.pth.tar")
            best_acc = valacc
        logging.info(f'best acc:{best_acc}')

        if epoch % 100 == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }
            torch.save(state, f"{args.ckpt_dir}/{logger_name}_epoch_{epoch}.pth.tar")

if __name__ == '__main__':
    main(args)
