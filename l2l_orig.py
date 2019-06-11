import torch
import argparse
import os
import pandas as pd
import torch.optim as optim
import numpy as np
import copy
import datetime
import time
from torch.utils.data import DataLoader
from utils_main import get_directory_l2l, get_dataset, get_hyper_parameters, get_need_parameters, N_samples_Sampler, \
                       Triplet_Loss, save_best_checkpoint, train_L2L_1, LSTM_Optimizer, MetaModel, MetaOptimizer, test_ML, train_L2L_1_BOTH, test_both
from vgg import VGG_ORIG as VGG

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch',    type=int,   default=1)
parser.add_argument('--num_epochs',     type=int,   default=100)  # Number of Epochs
parser.add_argument('--num_workers',    type=int,   default=4)  # Number of Workers
parser.add_argument('--batch_size',     type=int,   default=500)  # Batch Size
parser.add_argument('--dataset',        type=str,   default='cifar10', help='The Dataset to be used --- mnist|cifar10|fashion')  # Dataset
parser.add_argument('--momentum',       type=float, default=0.9)  # Momentum for SGD

parser.add_argument('--use_cuda',       type=int,   default=1)  # use cuda
parser.add_argument('--experiment',     type=str,   default="Deep Metric Clustering")  # Deep Metric Learning
parser.add_argument('--seed',           type=int,   default=123)
parser.add_argument('--continue',       type=int,   default=0)  # continue from where it is left
parser.add_argument('--resume',         type=str,   default='')  # Resume from where we left from before
parser.add_argument('--restore_id',     type=int,   default=-1)  # lambda3
parser.add_argument('--margin',         type=float, default=0.5)

parser.add_argument('--augment',        type=int,   default=0)  # [0, 1]
parser.add_argument('--type_exp',       type=str,   default='ml')  # [ml, classification, both]
parser.add_argument('--hyper',          type=str,   default=None)  # [R1, R2, R3, R4]
parser.add_argument('--optimizer',      type=str,   default='rmsprop', help='RMSProp|Adam')
parser.add_argument('--need',           type=int,   default=1)
parser.add_argument('--list',           type=list,  default=['fc2'])
'''
##----------------------------------------------------------------------------------------------------------------------
'''
parser.add_argument('--divide_by_N', action='store_true')
parser.add_argument('--clamp',       action='store_true')
parser.add_argument('--norm',        action='store_true')
parser.add_argument('--cat',         action='store_true')
'''
##----------------------------------------------------------------------------------------------------------------------
'''
parser.add_argument('--pre_process',    type=bool,  default=True)
parser.add_argument('--hidden_size',    type=str,   default='30,20,20')#)#, 20])
parser.add_argument('--unroll_steps',   type=int,   default=5)
parser.add_argument('--clamped_value',  type=float, default=None)
parser.add_argument('--dim',            type=int,   default=-1)
parser.add_argument('--p',              type=int,   default=2)
'''
#-----------------------------------------------------------------------------------------------------------------------
'''
config = parser.parse_args()
torch.manual_seed(config.seed)
H = config.hidden_size.split(',')
L = []
for e in H:
    L.append(int(e))
config.hidden_size = L
if torch.cuda.is_available() and config.use_cuda == 1:
    config.cuda = 1
else:
    config.cuda = 0

if os.path.exists('/flush2/roy030'):
    data_dir     = '/flush2/roy030/Data/ML'
    main_res_dir = '/flush1/roy030/NIPS/L2L-VGG9-ORIG/L2L-Results'
elif os.path.exists('/OSM/CBR/D61_RCV/students/roy030'):
    data_dir     = '/OSM/CBR/D61_RCV/students/roy030'
    main_res_dir = '/OSM/CBR/D61_RCV/students/roy030/NIPS/L2L-VGG9-ORIG/L2L-Results'
else:
    data_dir     = '/home/roy030/Desktop/CVPR-Clustering'
    main_res_dir = os.path.join(data_dir, 'NIPS/L2L-VGG9-ORIG/L2L-Results')

if config.need is not None:
    print "USING NEED NEED NEED NEED"
    config.hyper = get_need_parameters(need=config.need)

num_clusters, input_channels, train_dataset, test_dataset, \
            train_size, test_size, input_dim, res_dir = get_dataset(config.dataset.lower(), data_dir, config.augment)
lr, wd       = get_hyper_parameters(config.hyper, config.optimizer)
config.lr    = lr
config.wd    = wd
kwargs       = {'num_workers': config.num_workers, 'pin_memory': True} if config.cuda else {}
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
print "Creating the Train Loader for the ML experiment"
if config.dataset.lower() == 'stl10' or config.dataset.lower() == 'stl10-reduced':
    labels = train_dataset.labels
elif config.dataset.lower() == 'cifar100_20':
    labels = train_dataset.train_coarse_labels
else:
    labels = train_dataset.train_labels
print np.asarray(labels).max()

if config.dataset.lower() == 'mnist':
        N_samples_Sampler = N_samples_Sampler(batch_size=config.batch_size, labels=labels.numpy())
else:
        N_samples_Sampler = N_samples_Sampler(batch_size=config.batch_size, labels=np.asarray(labels))

train_loader1 = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, sampler=N_samples_Sampler, **kwargs)
Net           = VGG( dataset=config.dataset.lower(), vgg_name='VGG9', list=config.list,
		     param_list=['fc2.0.weight', 'fc2.0.bias'] )
if config.pre_process:
    print "PREPROCESSING OF INPUTS WILL BE DONE DONE DONE DONE"
else:
    print "NO PREPROCESSING OF INPUTS :( :( :( :("

if config.cat:
    print "CONCATENATION CONCATENATION CONCATENATION CONCATENATION"
    print "CONCATENATION OF INPUTS CONCATENATION OF INPUTS"
    LSTM    = LSTM_Optimizer(hidden_size = config.hidden_size, pre_process = config.pre_process, cat = True, use_cuda=config.cuda)
    res_dir = os.path.join('optimizer-'+ str(len(config.hidden_size))+ '-L2L-CAT', res_dir)
else:
    print "WITHOUT CONCATENATION OF INPUTS"
    LSTM    = LSTM_Optimizer(hidden_size=config.hidden_size, pre_process=config.pre_process, cat=False, use_cuda=config.cuda)
    res_dir = os.path.join('optimizer-' + str(len(config.hidden_size)) + '-L2L', res_dir)

if config.cuda:
    Net  = Net.cuda()
    LSTM = LSTM.cuda()
model_ml   = copy.deepcopy(Net)
lstm       = copy.deepcopy(LSTM)
meta_model = copy.deepcopy(model_ml)
print model_ml
del Net
del LSTM
n_params             = model_ml.num_params
main_res_dir, config = get_directory_l2l(main_res_dir=main_res_dir, res_dir=res_dir, config=config)
save_dir             = os.path.join(main_res_dir, config.optimizer.upper() + '-' + config.hyper)
print "Saving in the Directory", save_dir
if not os.path.exists(save_dir):
    print 'Creating the new Directory'
    os.makedirs(save_dir)
meta_optimizer     = MetaOptimizer(MetaModel(meta_model), lstms=lstm, n_params=n_params, hidden_size=config.hidden_size,config=config, use_cuda=config.cuda)
if config.cuda:
    meta_optimizer = meta_optimizer.cuda()

l2l_layers               = model_ml.list
default_optimizers_names = []
params                   = []
for name, module in model_ml.named_children():
    classname = module.__class__.__name__
    classname = classname.lower()
    if name not in l2l_layers and classname in ['sequential']:
        params += list(module.parameters())
        default_optimizers_names.append(name)
    elif name in l2l_layers:
        for child in module.modules():
            childname = child.__class__.__name__
            childname = childname.lower()
            if childname.lower() in ['conv1d', 'conv2d', 'batchnorm1d', 'batchnorm2d']:
                params += list(child.parameters())
                default_optimizers_names.append(name+'-'+childname)

if len(params)!=0:
    if config.optimizer.lower() == 'rmsprop':
        print "INSIDE MAIN using the RMSPROP optimizer"
        optimizer_ml = optim.RMSprop(params=params, lr=lr, weight_decay=wd)
        scheduler_ml = optim.lr_scheduler.StepLR(optimizer=optimizer_ml, step_size=50)
    elif config.optimizer.lower() == 'adam':
        print "INSIDE MAIN using the ADAM optimizer"
        optimizer_ml = optim.Adam(params, lr=lr, weight_decay=wd)
        scheduler_ml = optim.lr_scheduler.StepLR(optimizer=optimizer_ml, step_size=50)
    elif config.optimizer.lower() == 'sgd':
        print "INSIDE MAIN using the SGD optimizer"
        optimizer_ml = optim.SGD(params, lr=lr, weight_decay=wd, momentum=config.momentum)
        scheduler_ml = optim.lr_scheduler.StepLR(optimizer=optimizer_ml, step_size=50)

lstm_params = []
for i in range(len(config.hidden_size)):
    lstm_params += list(meta_optimizer.lstms.lstms[i].parameters())
lstm_params += list(meta_optimizer.lstms.lin.parameters())

if config.optimizer.lower() == 'rmsprop':
    print "INSIDE MAIN using the RMSPROP optimizer for LSTM"
    optimizer_L = optim.RMSprop(params=lstm_params, lr=config.lr, weight_decay=config.wd)
    scheduler_L = optim.lr_scheduler.StepLR(optimizer=optimizer_L, step_size=50)
elif config.optimizer.lower() == 'adam':
    print "INSIDE MAIN using the ADAM optimizer for LSTM"
    optimizer_L = optim.Adam(params=lstm_params, lr=config.lr, weight_decay=config.wd)
    scheduler_L = optim.lr_scheduler.StepLR(optimizer=optimizer_L, step_size=50)
elif config.optimizer.lower() == 'sgd':
    print "INSIDE MAIN using the SGD  optimizer for LSTM"
    optimizer_L = optim.SGD(params=lstm_params, lr=config.lr, weight_decay=config.wd, momentum=config.momentum)
    scheduler_L = optim.lr_scheduler.StepLR(optimizer=optimizer_L, step_size=50)

print "FUCKING ,,,,,,,,,,,, ", l2l_layers, default_optimizers_names
best_model       = model_ml
best_lstm        = lstm
best_optimizer   = optimizer_ml
best_scheduler   = scheduler_ml
best_optimizer_L = optimizer_L
best_scheduler_L = scheduler_L

results        = pd.DataFrame(index=np.arange(config.num_epochs), columns={'loss_triplet_lstm1', 'test_acc_triplet_lstm1'})
best_epochs    = 0
best_top_1s    = 0.0
top1_ml_lstm1  = test_ML(model=model_ml, train_loader=train_loader, test_loader=test_loader, config=config)
best_top_1s    = top1_ml_lstm1
trip_criterion = Triplet_Loss(margin=config.margin, cuda=config.cuda)

print "The top1 accuracies for the models are before Training "
print "VGG9_Triplet_LSTM1 : {:.2f}".format(best_top_1s)
train_time  = 0
start_time  = time.time()

for epoch in range(config.start_epoch, config.num_epochs + 1):
    print "-------------------------------- EPOCH {} ------------------------------------------".format(epoch)
    start_train_time = time.time()
    loss = train_L2L_1(epoch = epoch, model = model_ml, train_loader = train_loader, meta_optimizer = meta_optimizer,
                       criterion = trip_criterion, optimizer = optimizer_ml, scheduler = scheduler_ml, optimizer_L = optimizer_L,
                       scheduler_L = scheduler_L, config=config)
    train_time   += round(time.time() - start_train_time)
    top1_ml_lstm1 = test_ML(model=model_ml, train_loader=train_loader, test_loader=test_loader, config=config)
    epoch_time    = round(time.time() - start_train_time)
    epoch_time    = str(datetime.timedelta(seconds=epoch_time))
    print "The top1 accuracies for the models are:"
    print "VGG9_Triplet_LSTM1   LOSS:: {:.4f}  TOP1-ACC:: {:.2f}".format(loss, top1_ml_lstm1)
    print "Epoch Total Time is (h:m:s) : {}".format(epoch_time)
    i = 0
    if top1_ml_lstm1 > best_top_1s:
        best_top_1s      = top1_ml_lstm1
        best_epochs      = epoch
        best_model       = model_ml
        best_lstm        = lstm
        best_optimizer   = optimizer_ml
        best_scheduler   = scheduler_ml
        best_optimizer_L = optimizer_L
        best_scheduler_L = scheduler_L
        print "This is best epoch till now {}".format(epoch)
    print "TILL Now the Best Epoch is:: {}  best_top_1:: {:.2f}".format(best_epochs, best_top_1s.item())
    i += 1
    assert (i == 1)
    results.loc[epoch] = pd.Series({'loss_triplet_lstm1':loss, 'test_acc_triplet_lstm1':top1_ml_lstm1})
    results.to_csv(os.path.join(save_dir, 'results.csv'), columns=['loss_triplet_lstm1', 'test_acc_triplet_lstm1'])

elapsed    = round(time.time() - start_time)
elapsed    = str(datetime.timedelta(seconds=elapsed))
train_time = str(datetime.timedelta(seconds=train_time))
results.to_csv(os.path.join(save_dir, 'results.csv'), columns=['test_acc_triplet_lstm1', 'loss_triplet_lstm1'])
print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
print "-------------------------------- BEST ------------------------------------------"
print "VGG9_Triplet_LSTM1   ----> BEST_epoch::{}  BEST_top_1::{:.2f}".format(best_epochs, best_top_1s)
save_best_checkpoint(state={'best_top_1s':best_top_1s, 'best_epochs':best_epochs, 'best_model':best_model,
            'best_lstm':best_lstm, 'best_optimizer':best_optimizer, 'best_scheduler':best_scheduler,
            'best_optimizer_L':best_optimizer_L, 'best_scheduler_L':best_scheduler_L}, save_dir=save_dir)

'''
from utils_main import Distribution_Loss
criterion = Distribution_Loss(t=config.margin)
data           = torch.randn(10, 3, 32, 32)
labels         = torch.LongTensor(10).random_(4)
processed_data = 0
batch_idx = 0
meta_optimizer.reset_lstm(keep_states= (batch_idx > 0), model=model_ml)
loss_sum = 0
from utils_main import criterion_xent
for e in range(100000):
    batch_idx = e
    N = data.size(0)
    processed_data += N
    if config.cuda:
        data = data.cuda(); labels = labels.cuda()
#     ## -----------------------------------------------------------------------------------------------------------------
    output0, outputS = model_ml.forward(data)
    loss = criterion(features=output0, classes=labels)# + criterion_xent( outputS, labels)

    model_ml.zero_grad()
    optimizer_ml.zero_grad()
    loss.backward(retain_graph=True)

    # init = {}
    # for name, params in model_ml.named_parameters():
    #     init[name] = params.clone()

    meta_model = meta_optimizer.meta_update(model_with_grads=model_ml, num_samples=N, config=config)

    # final1 = {}
    # for name, params in model_ml.named_parameters():
    #     final1[name] = params.clone()


    model_ml.zero_grad()
    optimizer_ml.zero_grad()
    loss.backward()
    if optimizer_ml is not None :
        optimizer_ml.step()

        # final2 = {}
        # for name, params in model_ml.named_parameters():
        #     final2[name] = params.clone()
        #
        # for k in init.keys():
        #     print "HELLO --->", k, "--->", torch.abs(final1[k] - final2[k]).sum().item(), \
        #         "--->", torch.abs(final1[k] - init[k]).sum().item(), "--->", torch.abs(final2[k] - init[k]).sum().item()
    #
    #
    output0_meta, outputS_meta = meta_model(data)
    new_loss = criterion(output0_meta, labels) + criterion_xent(outputS_meta, labels)
    loss_sum += new_loss
    if (batch_idx+1)% 10 == 0:
        meta_optimizer.zero_grad()
        loss_sum.backward()
        loss_sum = torch.zeros(1)
        if config.cuda:
            loss_sum = loss_sum.cuda()
        l1 = meta_optimizer.lstms.lin.weight.data.clone()
        # l2 = meta_optimizer.lstms.layer1.weight_hh.data.clone()
        # l3 = meta_optimizer.lstms.layer1.weight_ih.data.clone()
        optimizer_L.step()
        m1 = meta_optimizer.lstms.lin.weight.data.clone()
        # m2 = meta_optimizer.lstms.layer1.weight_hh.data.clone()
        # m3 = meta_optimizer.lstms.layer1.weight_ih.data.clone()
        print (batch_idx+1), " HELLO --> First ", torch.abs(l1-m1).sum().item()#, torch.abs(l2-m2).sum().item(), torch.abs(l3-m3).sum().item()
#
'''

'''
res_dir  = 'optimizer-'+ str(len(config.hidden_size))+ '-L2L-BOTH'
LSTM1    = LSTM_Optimizer(hidden_size = config.hidden_size, pre_process = config.pre_process, cat = False, use_cuda=config.cuda)
LSTM2    = LSTM_Optimizer(hidden_size=config.hidden_size, pre_process=config.pre_process, cat=True, use_cuda=config.cuda)

if config.cuda:
    Net   = Net.cuda()
    LSTM1 = LSTM1.cuda()
    LSTM2 = LSTM2.cuda()

model1          = copy.deepcopy(Net)
lstm1           = copy.deepcopy(LSTM1)
meta_model1     = copy.deepcopy(model1)
n_params1       = model1.num_params
meta_optimizer1 = MetaOptimizer(MetaModel(meta_model1), lstms=lstm1, n_params=n_params1, hidden_size=config.hidden_size,
                               config=config, use_cuda=config.cuda)
if config.cuda:
    meta_optimizer1 = meta_optimizer1.cuda()

l2l_layers1               = model1.list
default_optimizers_names1 = []
params1                   = []
for name, module in model1.named_children():
    classname = module.__class__.__name__
    classname = classname.lower()
    if name not in l2l_layers1 and classname in ['sequential']:
        params1 += list(module.parameters())
        default_optimizers_names1.append(name)
    elif name in l2l_layers1:
        for child in module.modules():
            childname = child.__class__.__name__
            childname = childname.lower()
            if childname.lower() in ['conv1d', 'conv2d', 'batchnorm1d', 'batchnorm2d']:
                params1 += list(child.parameters())
                default_optimizers_names1.append(name+'-'+childname)

if len(params1)!=0:
    if config.optimizer.lower() == 'rmsprop':
        print "INSIDE MAIN using the RMSPROP optimizer"
        optimizer1 = optim.RMSprop(params=params1, lr=lr, weight_decay=wd)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=50)
    elif config.optimizer.lower() == 'adam':
        print "INSIDE MAIN using the ADAM optimizer"
        optimizer1 = optim.Adam(params1, lr=lr, weight_decay=wd)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=50)
    elif config.optimizer.lower() == 'sgd':
        print "INSIDE MAIN using the SGD optimizer"
        optimizer1 = optim.SGD(params1, lr=lr, weight_decay=wd, momentum=config.momentum)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=50)
print "THE LAYERS WHICH WILL BE HANDLED BY DEFAULT OPTIMIZER is ", default_optimizers_names1

if config.optimizer.lower() == 'rmsprop':
    print "INSIDE MAIN using the RMSPROP optimizer for LSTM"
    optimizer_L1 = optim.RMSprop(meta_optimizer1.lstms.parameters(), lr=lr, weight_decay=wd)
    scheduler_L1 = optim.lr_scheduler.StepLR(optimizer=optimizer_L1, step_size=50)
elif config.optimizer.lower() == 'adam':
    print "INSIDE MAIN using the ADAM optimizer for LSTM"
    optimizer_L1 = optim.Adam(meta_optimizer1.lstms.parameters(), lr=lr, weight_decay=wd)
    scheduler_L1 = optim.lr_scheduler.StepLR(optimizer=optimizer_L1, step_size=50)
elif config.optimizer.lower() == 'sgd':
    print "INSIDE MAIN using the SGD  optimizer for LSTM"
    optimizer_L1 = optim.SGD(meta_optimizer1.lstms.parameters(), lr=lr, weight_decay=wd, momentum=config.momentum)
    scheduler_L1 = optim.lr_scheduler.StepLR(optimizer=optimizer_L1, step_size=50)


model2          = copy.deepcopy(Net)
lstm2           = copy.deepcopy(LSTM2)
meta_model2     = copy.deepcopy(model2)
n_params2       = model2.num_params
meta_optimizer2 = MetaOptimizer(MetaModel(meta_model2), lstms=lstm2, n_params=n_params2, hidden_size=config.hidden_size,
                               config=config, use_cuda=config.cuda)

l2l_layers2               = model2.list
default_optimizers_names2 = []
params2                   = []
for name, module in model2.named_children():
    classname = module.__class__.__name__
    classname = classname.lower()
    if name not in l2l_layers2 and classname in ['sequential']:
        params2 += list(module.parameters())
        default_optimizers_names2.append(name)
    elif name in l2l_layers2:
        for child in module.modules():
            childname = child.__class__.__name__
            childname = childname.lower()
            if childname.lower() in ['conv1d', 'conv2d', 'batchnorm1d', 'batchnorm2d']:
                params2 += list(child.parameters())
                default_optimizers_names2.append(name+'-'+childname)

if len(params2)!=0:
    if config.optimizer.lower() == 'rmsprop':
        print "INSIDE MAIN using the RMSPROP optimizer"
        optimizer2 = optim.RMSprop(params=params2, lr=lr, weight_decay=wd)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer=optimizer2, step_size=50)
    elif config.optimizer.lower() == 'adam':
        print "INSIDE MAIN using the ADAM optimizer"
        optimizer2 = optim.Adam(params2, lr=lr, weight_decay=wd)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer=optimizer2, step_size=50)
    elif config.optimizer.lower() == 'sgd':
        print "INSIDE MAIN using the SGD optimizer"
        optimizer2 = optim.SGD(params2, lr=lr, weight_decay=wd, momentum=config.momentum)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer=optimizer2, step_size=50)
print "THE LAYERS WHICH WILL BE HANDLED BY DEFAULT OPTIMIZER is ", default_optimizers_names1

if config.cuda:
    meta_optimizer2 = meta_optimizer2.cuda()

if config.optimizer.lower() == 'rmsprop':
    print "INSIDE MAIN using the RMSPROP optimizer for LSTM"
    optimizer_L2 = optim.RMSprop(meta_optimizer2.lstms.parameters(), lr=lr, weight_decay=wd)
    scheduler_L2 = optim.lr_scheduler.StepLR(optimizer=optimizer_L2, step_size=50)
elif config.optimizer.lower() == 'adam':
    print "INSIDE MAIN using the ADAM optimizer for LSTM"
    optimizer_L2 = optim.Adam(meta_optimizer2.lstms.parameters(), lr=lr, weight_decay=wd)
    scheduler_L2 = optim.lr_scheduler.StepLR(optimizer=optimizer_L2, step_size=50)
elif config.optimizer.lower() == 'sgd':
    print "INSIDE MAIN using the SGD  optimizer for LSTM"
    optimizer_L2 = optim.SGD(meta_optimizer2.lstms.parameters(), lr=lr, weight_decay=wd, momentum=config.momentum)
    scheduler_L2 = optim.lr_scheduler.StepLR(optimizer=optimizer_L2, step_size=50)


del Net
del LSTM1
del LSTM2

main_res_dir, config = get_directory_l2l(main_res_dir, res_dir, config)
save_dir     = main_res_dir
print "Saving in the Directory", save_dir
if not os.path.exists(save_dir):
    print 'Creating the new Directory'
    os.makedirs(save_dir)
criterion         = Triplet_Loss(margin=config.margin, cuda=config.cuda)
best_model1       = model1
best_lstm1        = lstm1
best_optimizer1   = optimizer1
best_scheduler1   = scheduler1
best_optimizer_L1 = optimizer_L1
best_scheduler_L1 = scheduler_L1
best_meta_optimizers1 = meta_optimizer1

best_model2           = model2
best_lstm2            = lstm2
best_optimizer2       = optimizer2
best_scheduler2       = scheduler2
best_optimizer_L2     = optimizer_L2
best_scheduler_L2     = scheduler_L2
best_meta_optimizers2 = meta_optimizer2


models          = [model1,          model2]
lstms           = [lstm1,           lstm2]
meta_models     = [meta_model1,     meta_model2]
optimizers      = [optimizer1,      optimizer2]
schedulers      = [scheduler1,      scheduler2]
lstm_optimizers = [optimizer_L1,    optimizer_L2]
lstm_schedulers = [scheduler_L1,    scheduler_L2]
meta_optimizers = [meta_optimizer1, meta_optimizer2]


results          = pd.DataFrame(index = np.arange(config.num_epochs), columns = {'loss1', 'test_lstm1', 'loss1-cat', 'test_lstm1-cat'})
best_epochs    = torch.zeros(2).int()
best_top_1s    = torch.zeros(2)
top1, top1_cat = test_both( model = models, train_loader = train_loader, test_loader = test_loader, config = config )
best_top_1s[0] = top1; best_top_1s[1] = top1_cat

print "The top1 accuracies for the models are before Training "
print "Model_Triplet_LSTM1 :: {:.2f}    CAT :: {:.2f}".format(top1, top1_cat)
train_time  = 0
start_time  = time.time()

for epoch in range(config.start_epoch, config.num_epochs+1):
    print "-------------------------------- EPOCH {} ------------------------------------------".format(epoch)
    start_train_time = time.time()
    loss, loss_cat = train_L2L_1_BOTH(epoch = epoch, model = models, train_loader = train_loader, meta_optimizer = meta_optimizers,
                                criterion = criterion, optimizer = optimizers, scheduler = schedulers, optimizer_L = lstm_optimizers,
                                scheduler_L = lstm_schedulers, config=config)

    train_time += round(time.time() - start_train_time)
    top1, top1_cat = test_both(model=models, train_loader=train_loader, test_loader=test_loader, config=config)
    epoch_time = round(time.time() - start_train_time)
    epoch_time = str(datetime.timedelta(seconds=epoch_time))
    print "The top1 accuracies for the models are before Training "
    print "Model_Triplet_LSTM1 :: {:.2f}    CAT :: {:.2f}".format(top1, top1_cat)
    print "Epoch Total Time is (h:m:s): {}".format(epoch_time)

    if top1 > best_top_1s[0]:
        best_top_1s[0]       = top1
        best_epochs[0]    = epoch
        best_model1       = model1
        best_lstm1        = lstm1
        best_optimizer1   = optimizer1
        best_scheduler1   = scheduler1
        best_optimizer_L1 = optimizer_L1
        best_scheduler_L1 = scheduler_L1
        print "For MODEL WITHOUT CONCATENATION --- This is best epoch till now {}".format(epoch)

    if top1_cat > best_top_1s[1]:
        best_top_1s[1]    = top1_cat
        best_epochs[1]    = epoch
        best_model2       = model2
        best_lstm2        = lstm2
        best_optimizer2   = optimizer2
        best_scheduler2   = scheduler2
        best_optimizer_L2 = optimizer_L2
        best_scheduler_L2 = scheduler_L2
        print "For MODEL WITH CONCATENATION --- This is best epoch till now {}".format(epoch)

    print "TILL Now the Best Epoch FOR MODEL-1::{}({:.2f})    MODEL-1-CAT::{}({:.2f})"\
        .format(best_epochs[0], best_top_1s[0].item(), best_epochs[1], best_top_1s[1].item())

    results.loc[epoch] = pd.Series({'loss1':loss, 'test_lstm1':top1, 'loss1-cat':loss_cat, 'test_lstm1-cat':top1_cat})

elapsed    = round(time.time() - start_time)
elapsed    = str(datetime.timedelta(seconds=elapsed))
train_time = str(datetime.timedelta(seconds=train_time))
results.to_csv(os.path.join(save_dir, 'results.csv'), columns=['test_lstm1', 'test_lstm1-cat', 'loss1', 'loss1-cat'])

print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
print "-------------------------------- BEST ------------------------------------------"
print "Model_Triplet_LSTM1      ----> best_epoch::{}  best_top_1::{:.2f}".format(best_epochs[0], best_top_1s[0].item())
print "Model_Triplet_LSTM1-CAT  ----> best_epoch::{}  best_top_1::{:.2f}".format(best_epochs[1], best_top_1s[1].item())
save_best_checkpoint(state={'best_top_1s':best_top_1s, 'best_epochs':best_epochs, 'best_model1':best_model1, 'best_lstm1':best_lstm1,
                            'best_optimizer1':best_optimizer1, 'best_scheduler':best_scheduler1, 'best_optimizer_L1':best_optimizer_L1,
                            'best_scheduler_L1':best_scheduler_L1, 'best_model2':best_model2, 'best_lstm2':best_lstm2,
                            'best_optimizer2':best_optimizer2, 'best_scheduler2':best_scheduler2, 'best_optimizer_L2':best_optimizer_L2,
                            'best_scheduler_L2':best_scheduler_L2 },  save_dir=save_dir)
'''
