import math
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.datasets import datasets
from utilis.datasets import Collate_function
from torch.utils.tensorboard import SummaryWriter

import models
from ops.config import parser
from training.schedule import lr_setter
from training.train import train
from training.validate import validate
from utilis.meters import AverageMeter
from utilis.saving import save_checkpoint

from transformers import AutoConfig, AutoTokenizer, AdamW
from models.model_slabt import AutoModelForSlabt
from info_regularizer import (CLUB, InfoNCE)

best_acc1 = 0



def main():
    args = parser.parse_args()

    args.classes_num = 3

    args.log_path = os.path.join(args.log_base, args.dataset, "log.txt")

    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # <--- args.distributed = False
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.pretrained:
        config = AutoConfig.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSlabt('bert-base-uncased', config=config, args=args, cls_num=args.classes_num)
    else:
        pass

    num_ftrs = model.fc1.in_features
    model.fc1 = nn.Linear(num_ftrs, args.classes_num)
    nn.init.xavier_uniform_(model.fc1.weight, .1)
    nn.init.constant_(model.fc1.bias, 0.)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None: # <--- here we go
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    hidden_size = model.bert_config.hidden_size
    mi_upper_estimator = CLUB(hidden_size, hidden_size, beta=args.beta).cuda(args.gpu)
    mi_estimator = InfoNCE(hidden_size, hidden_size).cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_train = nn.CrossEntropyLoss(reduce=False).cuda(args.gpu)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # /root/slabt/dataset/ + MNLI/ + 
    traindir = os.path.join(args.data, args.dataset, 'mnli_trainset.tsv')
    valdir = os.path.join(args.data, args.dataset, 'mnli_valset.tsv')
    testdir = os.path.join(args.data, args.dataset, 'mnli_testset.tsv')
    
    # /root/slabt/dataset/ + HANS/ + 
    test2dir = os.path.join(args.data, args.sub_dataset, 'hans_testset.tsv')


    labels_type = ["contradiction", "entailment", "neutral"]


    log_dir = os.path.dirname(args.log_path)
    print('tensorboard dir {}'.format(log_dir))
    tensor_writer = SummaryWriter(log_dir)

    if args.evaluate:

        val_dataset = datasets(valdir, tokenizer, labels_type, args)
        test_dataset = datasets(testdir, tokenizer, labels_type, args)
        test2_dataset = datasets(test2dir, tokenizer, labels_type, args)


        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())
        test2_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())



        validate(val_loader, model, criterion, 0, False, args, tensor_writer, datasetname='MNLI')
        validate(test_loader, model, criterion, 0, True, args, tensor_writer, datasetname='MNLI')
        validate(test2_loader, model, criterion, 0, True, args, tensor_writer, datasetname='HANS')

        return


    train_dataset = datasets(traindir, tokenizer, labels_type, args)
    val_dataset = datasets(valdir, tokenizer, labels_type, args)
    test_dataset = datasets(testdir, tokenizer, labels_type, args)
    test2_dataset = datasets(test2dir, tokenizer, labels_type, args)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    test2_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())
    
    print('\n*****train_dataset len is : {}'.format(len(train_dataset)))
    print('\n*****val_dataset len is : {}'.format(len(val_dataset)))
    print('\n*****test_dataset len is : {}'.format(len(test_dataset)))


    # begin to train
    for epoch in range(args.start_epoch, args.epochs):

        train(train_loader, model, criterion_train, optimizer, epoch, args, tensor_writer=tensor_writer, mi_upper_estimator=mi_upper_estimator, dow=mi_estimator)

        val_acc1 = validate(val_loader, model, criterion, epoch, False, args, tensor_writer, datasetname='MNLI')
        acc1 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='MNLI')
        acc2 = validate(test2_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='HANS')

        is_best = val_acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # true
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print('Saving...')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.log_path, epoch)


if __name__ == '__main__':
    main()
