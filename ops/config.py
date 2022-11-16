import argparse

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-data', metavar='DIR', default='/root/slabt/dataset',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='bert',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: bert)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=8, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')


parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--cos', '--cosine_lr', default=1, type=int,
                    metavar='COS', help='lr decay by decay', dest='cos')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')


parser.add_argument('--beta', default=0.0, type=float,
                    help='hyperparam of Info upper bound')


parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=777, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log_base',
                    default='./result_s777lr5e5epo8lrbl27', type=str, metavar='PATH',
                    help='path to save logs (default: none)')

# for number of fourier spaces
parser.add_argument ('--num_f', type=int, default=1, help = 'number of fourier spaces')

parser.add_argument ('--sample_rate', type=float, default=1.0, help = 'sample ratio of the features involved in balancing')
parser.add_argument ('--lrbl', type = float, default = 2.7, help = 'learning rate of balance')

parser.add_argument ('--lambdap', type = float, default = 70.0, help = 'weight decay for weight1 ')
parser.add_argument ('--lambdapre', type = float, default = 1, help = 'weight for pre_weight1 ')

parser.add_argument ('--epochb', type = int, default = 20, help = 'number of epochs to balance')
parser.add_argument ('--epochp', type = int, default = 0, help = 'number of epochs to pretrain')

# debug n_feature from 128 to 64    ablustr
parser.add_argument ('--n_feature', type=int, default=32, help = 'number of pre-saved features')
# debug feature_dim from 512 to 768    ablustr
parser.add_argument ('--feature_dim', type=int, default=768, help = 'the dim of each feature')

parser.add_argument ('--lrwarmup_epo', type=int, default=0, help = 'the dim of each feature')
parser.add_argument ('--lrwarmup_decay', type=int, default=0.1, help = 'the dim of each feature')

parser.add_argument ('--n_levels', type=int, default=1, help = 'number of global table levels')

# for expectation
parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')

# for first step
parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

# for pow
parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')

# for lr decay epochs
parser.add_argument ('--epochs_decay', type=list, default=[24, 30], help = 'weight lambda for second order moment loss')

parser.add_argument ('--classes_num', type=int, default=5, help = 'number of epoch for lambda to decay')

parser.add_argument ('--dataset', type=str, default="MNLI", help = '')
parser.add_argument ('--sub_dataset', type=str, default="HANS", help = '')
parser.add_argument ('--gray_scale', type=float, default=0.1, help = 'weight lambda for second order moment loss')

parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--concat', type=int, default=1, help='sum or concat')
parser.add_argument('--min_scale', type=float, default=0.8, help='')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')
