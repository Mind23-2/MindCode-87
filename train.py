
import os
import time
from tqdm import tqdm
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
from mindspore.train.callback import LearningRateScheduler, TimeMonitor,LossMonitor

from models import TSN_light
from utils import AverageMeter
from options import args
from dataset import Create_UCF101_Dataset

args = {}
args['device_target'] = "Ascend"
args['is_distributed'] = False
args['device_num'] = 1

device_id = int(os.getenv('DEVICE_ID', '7'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args['device_target'], save_graphs=False, device_id=device_id)
# context.PYNATIVE_MODE

args['rank'] = 0

if(args['is_distributed']):
    if args['device_target'] == "Ascend":
        init()
    else:
        init("nccl")
    
    args['rank'] = get_rank()
    args['group_size'] = get_group_size()
    device_num = args['group_size']
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True)
else:
    pass

args['datapath'] = '/data/XDU/UCF101_hmdb51/UCF-101/'
args['datalist_path'] = 'datalist/ucf101/'
args['weight_savefile'] = 'save_model/tsn.model'
args['weight_save_path'] = 'save_model/'
args['batch_size_train'] = 256

args['ckpt_interval'] = 1
args['lr'] = 0.001
args['epoch'] = 45

args['dataset'] = 'ucf101'


def step_decay(lr, cur_step_num):
    step_per_lr = int(9537 / args['batch_size_train'] / args['device_num'])

    if(cur_step_num % step_per_lr == 0):
        lr = lr / 10
    
    return lr


if args['dataset'] == 'ucf101':
    num_class = 101
elif args['dataset'] == 'hmdb51':
    num_class = 51
elif args['dataset'] == 'kinetics':
    num_class = 400
else:
    raise ValueError('Unknown dataset ' + args['dataset'])

def train_model(num_classes, directory_dataset, batch_size = 256, num_epochs=45, save=True, savefile_weight="save_model/0-1_1.ckpt"):
    # initalize the ResNet 18 version of this model
    model = TSN_light(num_classes=num_classes)


    # check if there was a previously saved checkpoint
    if os.path.exists(savefile_weight):
        # loads the checkpoint
        param_dict = load_checkpoint(savefile_weight)
        
        load_param_into_net(model, param_dict)
        print("Reloading from previously saved checkpoint")
    else:
        print('Can not find exist check point file, this run will start with begining. ')


    ckpt_max_num = num_epochs
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args['ckpt_interval'],
                                       keep_checkpoint_max=ckpt_max_num)
    save_ckpt_path = os.path.join(args['weight_save_path'], 'ckpt_' + str(args['rank']) + '/')
    ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(args['rank']))
    cb_params = _InternalCallbackParam()
    cb_params.train_network = model
    cb_params.epoch_num = ckpt_max_num
    cb_params.cur_epoch_num = 1
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)
    
    loss_meter = AverageMeter('loss')
    epoch_current = 0
    epoch_last = 0
    step = 0
    t_end = time.time()

    time_cb = TimeMonitor(data_size=1)
    loss_cb = LossMonitor(per_print_times=1)
    callbacks = [time_cb, loss_cb, LearningRateScheduler(step_decay)]

    optimizer = ms.nn.SGD(model.trainable_params(), learning_rate=args['lr'], weight_decay=0.0)
    loss = nn.SoftmaxCrossEntropyWithLogits()
    model_train = Model(model, loss_fn = loss, optimizer=optimizer)
    
    dataset = Create_UCF101_Dataset(args['datapath'], args['datalist_path'], mode = 'train', batch_size = 10, shuffle=True)
    model_train.train(num_epochs, dataset, callbacks=callbacks)

    pass


if(__name__ == '__main__'):
    train_model(num_class,args['datapath'],args['batch_size_train'],args['epoch'], save=True, savefile_weight= args['weight_savefile'])
