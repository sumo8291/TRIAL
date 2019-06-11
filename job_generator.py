import itertools
import os
import argparse

parser      = argparse.ArgumentParser()
parser.add_argument('--main_file',  type=str, default='l2l_trip.py')
config     = parser.parse_args()
print config
files       = {'l2l_trip.py': 'L2L-TRIP', 'l2l_ang.py':'L2L-ANG-'}
batch_sizes = {'cifar100': [200, 300], 'cifar100_20': [100,200],  'cifar10': [100, 200], 'stl10-reduced': [100, 200]}

datasets    = ['cifar10', 'cifar100_20', 'cifar100','stl10-reduced']#]#
ES          = [64, 128, 256]
cat         = [0, 1]#, 1]
augment     = [0, 1]#, 1]
optimizers  = ['rmsprop', 'adam', 'sgd']
need        = [1]
hidden_size = [[30, 20, 20]]#
#                 #, 20], [30, 30, 20, 30], [30, 20, 20, 30],
#                 # [30, 40, 30, 30],[30, 20, 40, 40], [30, 20, 40, 20], [30, 20, 40, 20],
#                 # [30, 40, 30, 50], [30, 50, 40, 40], [30, 40, 40, 50], [30, 50, 40, 20] ]
# # hidden_size += [[10, 10], [10, 20], [10, 30], [10, 40],
# #                [20, 10], [20, 20], [20, 30], [20, 40],
# #                [30, 10], [30, 20], [30, 30], [30, 40],
# #                [40, 10], [40, 20], [40, 30], [40, 40]]
# # hidden_size  = []
# # hidden_size += list(itertools.product([20, 30, 40, 50] , repeat=4))
# # print len(hidden_size), hidden_size
exp_id  = 0
if config.main_file.lower() == 'l2l_trip.py':
    folder = 'Logs_L2L_ML_AGAIN'
    if not os.path.exists(folder):
        print 'Creating the new Log Directory=' + folder
        os.makedirs(folder)
    exp_id  = 0
    margins = [0.5, 1.0]
    for dataset in datasets:
        if dataset not in ['stl10-reduced', 'stl10']:
            parameters_list  = [[dataset], ES, hidden_size, cat, augment, margins, batch_sizes[dataset], optimizers, need]
            all_combinations = list(itertools.product(*parameters_list))
            for parameter_set in all_combinations:
                exp_id      += 1
                dataset_val     = parameter_set[0]
                ES_val          = parameter_set[1]
                hidden_size_val = parameter_set[2]
                cat_val         = parameter_set[3]
                augment_val     = parameter_set[4]
                margin_val      = parameter_set[5]
                batch_size_val  = parameter_set[6]
                optim_val       = parameter_set[7]
                need_val        = parameter_set[8]
                H, H1           = ','.join(str(e) for e in hidden_size_val), '-'.join(str(e) for e in hidden_size_val)
                H, H1           = "'" + H + "'", '[' + H1 + ']'
                if cat_val == 1:
                    extra = '  --cat '; added = '-CAT-'
                else:
                    extra = ' '; added = '-'
                job_name1 = files[config.main_file] + added + dataset_val + '-ES-' + str(ES_val) \
                            + '-BS-' + str(batch_size_val) + '-HID-' + H1  + '-AUG-' + str(augment_val) \
                            + '-Mar-' + str(margin_val) + '-' + optim_val.upper() + '-' + str(need_val)
                f = open(job_name1 + '.q', 'w')
                f.write('#!/bin/bash\n')
                f.write('#SBATCH --job-name=' + job_name1 + '\n')
                f.write('#SBATCH --time=06:00:00\n')
                f.write('#SBATCH --gres=gpu:1\n')
                f.write('#SBATCH --error=err_' + job_name1 + '.txt\n')
                f.write('#SBATCH --output=out_' + job_name1 + '.txt\n')
                f.write('#SBATCH --ntasks-per-node=1\n')
                f.write('#SBATCH --mem=12g\n')
                f.write('#SBATCH --mail-type=ALL\n')
                f.write('#SBATCH --mail-user=soumava.roy91@gmail.com\n')
                f.write('\n \n')
                f.write('module load python\n')
                f.write('module load pytorch/0.4.0-py27-cuda91\n')
                f.write('module load torchvision/0.2.1-py27\n')
                H2 = job_name1
                f.write('\n \n')
                f.write('python ' + config.main_file + '  --dataset ' + dataset_val + '  --augment ' + str(augment_val)
                        + '  --embedding_size ' + str(ES_val) + '  --hidden_size ' + H  + ' --margin  ' + str(margin_val)
                        + extra + '  --batch_size ' + str(batch_size_val) + '  --optimizer  ' + optim_val
                        + '  --need  ' + str(need_val) + ' > ' + folder + os.sep + H2 + '.log')
                # break
        else:
            parameters_list = [[dataset], ES, hidden_size, cat, augment, margins, batch_sizes[dataset]]
            all_combinations = list(itertools.product(*parameters_list))
            for parameter_set in all_combinations:
                exp_id         += 1
                dataset_val     = parameter_set[0]
                ES_val          = parameter_set[1]
                hidden_size_val = parameter_set[2]
                cat_val         = parameter_set[3]
                augment_val     = parameter_set[4]
                margin_val      = parameter_set[5]
                batch_size_val  = parameter_set[6]
                H, H1 = ','.join(str(e) for e in hidden_size_val), '-'.join(str(e) for e in hidden_size_val)
                H, H1 = "'" + H + "'", '[' + H1 + ']'
                if cat_val == 1:
                    extra = '  --cat '; added = '-CAT-'
                else:
                    extra = ' '; added = '-'
                job_name1 = files[config.main_file] + added + dataset_val + '-ES-' + str(ES_val) \
                            + '-BS-' + str(batch_size_val) + '-HID-' + H1 + '-AUG-' + str(augment_val) \
                            + '-Mar-' + str(margin_val)
                f = open(job_name1 + '.q', 'w')
                f.write('#!/bin/bash\n')
                f.write('#SBATCH --job-name=' + job_name1 + '\n')
                f.write('#SBATCH --time=06:00:00\n')
                f.write('#SBATCH --gres=gpu:1\n')
                f.write('#SBATCH --error=err_' + job_name1 + '.txt\n')
                f.write('#SBATCH --output=out_' + job_name1 + '.txt\n')
                f.write('#SBATCH --ntasks-per-node=1\n')
                f.write('#SBATCH --mem=12g\n')
                f.write('#SBATCH --mail-type=ALL\n')
                f.write('#SBATCH --mail-user=soumava.roy91@gmail.com\n')
                f.write('\n \n')
                f.write('module load python\n')
                f.write('module load pytorch/0.4.0-py27-cuda91\n')
                f.write('module load torchvision/0.2.1-py27\n')
                for optim_val in optimizers:
                    for need_val in need:
                        H2 = job_name1 + '-' + optim_val.upper() + '-' + str(need_val)
                        f.write('\n \n')
                        f.write('python ' + config.main_file + '  --dataset ' + dataset_val + '  --augment ' + str(augment_val)
                                + '  --embedding_size ' + str(ES_val) + '  --hidden_size ' + H + ' --margin  ' + str(margin_val)
                                + extra + '  --batch_size ' + str(batch_size_val) + '  --optimizer  ' + optim_val
                                + '  --need  ' + str(need_val) + ' > ' + folder + os.sep + H2 + '.log')
                # break
print exp_id


# parser                  = argparse.ArgumentParser()
# parser.add_argument('--main_file',  type=str, default='main_ang.py')
# config     = parser.parse_args()
# print config
# files      = {'l2l_ml.py': 'L2L-ML', 'main_trip.py': 'MAIN-TRIP',
#               'orig_ml.py': 'ORIG-ML', 'main_ang.py':'MAIN-ANG'}
# exp_id     = 0
#
# ES         = [64, 128, 256]
# augment    = [0, 1]
# optimizers = ['rmsprop', 'adam', 'sgd']
# need       = [1, 2]
# folder     = 'Logs_ML_BASELINE'
# if not os.path.exists(folder):
#     print 'Creating the new Log Directory=' + folder
#     os.makedirs(folder)
#
# if config.main_file.lower() == 'main_trip.py':
#  margin      = [0.5, 1.0]
#
#  for dataset in batch_sizes.keys():
#    print config.main_file.lower(), dataset
#    if dataset in ['cifar10', 'cifar100_20', 'cifar100']:
#     batch_size       = batch_sizes[dataset]
#     parameters_list  = [[dataset], ES, batch_size, margin, augment, optimizers]#, need]
#     all_combinations = list(itertools.product(*parameters_list))
#     for parameter_set in all_combinations:
#         exp_id      += 1
#         dataset_val    = parameter_set[0]
#         ES_val         = parameter_set[1]
#         batch_size_val = parameter_set[2]
#         margin_val     = parameter_set[3]
#         augment_val    = parameter_set[4]
#         optim_val      = parameter_set[5]
#         # need_val       = parameter_set[6]
#         job_name1      = files[config.main_file] + '-' + dataset_val + '-ES-' + str(ES_val) + '-BS-' + str(batch_size_val) \
#                          + '-Margin-' + str(margin_val) + '-AUG-' + str(augment_val) + '-' + optim_val# + '-' + str(need_val)
#         f = open(job_name1 + '.q', 'w')
#         f.write('#!/bin/bash\n')
#         f.write('#SBATCH --job-name=' + job_name1 + '\n')
#         f.write('#SBATCH --time=04:00:00\n')
#         f.write('#SBATCH --gres=gpu:1\n')
#         f.write('#SBATCH --error=err_' + job_name1 + '.txt\n')
#         f.write('#SBATCH --output=out_' + job_name1 + '.txt\n')
#         f.write('#SBATCH --ntasks=4\n')
#         # f.write('#SBATCH --ntasks-per-node=1\n')
#         f.write('#SBATCH --mem=12g\n')
#         f.write('#SBATCH --mail-type=ALL\n')
#         f.write('#SBATCH --mail-user=soumava.roy91@gmail.com\n')
#         f.write('\n \n')
#         f.write('module load parallel\n')
#         f.write('module load pytorch/0.4.0-py27-cuda91\n')
#         f.write('module load torchvision/0.2.1-py27\n')
#         H2 = job_name1
#         f.write('\n \n')
#         f.write(' seq 2 | parallel -j4 \' python ' + config.main_file + '  --dataset ' + dataset_val
#                 + '  --augment ' + str(augment_val) + '  --embedding_size ' + str(ES_val)
#                 + '  --batch_size ' + str(batch_size_val) + '  --optimizer '  + optim_val  + '  --margin ' + str(margin_val)
#                 + '  --need  {}  > ' + folder + os.sep + H2 + '-"{}".log\'')
#                 # + '  --need  ' + str(need_val) + ' > ' + folder + os.sep + H2 + '.log')
#         f.close()
#
#    # if dataset in ['stl10-reduced']:
#    #  batch_size       = batch_sizes[dataset]
#    #  parameters_list  = [[dataset], ES, batch_size, margin, augment]#, optimizers, need]
#    #  all_combinations = list(itertools.product(*parameters_list))
#    #  for parameter_set in all_combinations:
#    #      exp_id      += 1
#         # dataset_val    = parameter_set[0]
#         # ES_val         = parameter_set[1]
#         # batch_size_val = parameter_set[2]
#         # margin_val     = parameter_set[3]
#         # augment_val    = parameter_set[4]
#         # job_name1      = files[config.main_file] + '-' + dataset_val + '-ES-' + str(ES_val) + '-BS-' + str(batch_size_val) \
#         #                  + '-Margin-' + str(margin_val) + '-AUG-' + str(augment_val)
#         # f = open(job_name1 + '.q', 'w')
#         # f.write('#!/bin/bash\n')
#         # f.write('#SBATCH --job-name=' + job_name1 + '\n')
#         # f.write('#SBATCH --time=04:00:00\n')
#         # f.write('#SBATCH --gres=gpu:1\n')
#         # f.write('#SBATCH --error=err_' + job_name1 + '.txt\n')
#         # f.write('#SBATCH --output=out_' + job_name1 + '.txt\n')
#         # f.write('#SBATCH --ntasks-per-node=1\n')
#         # f.write('#SBATCH --mem=12g\n')
#         # f.write('#SBATCH --mail-type=ALL\n')
#         # f.write('#SBATCH --mail-user=soumava.roy91@gmail.com\n')
#         # f.write('\n \n')
#         # f.write('module load parallel\n')
#         # f.write('module load pytorch/0.4.0-py27-cuda91\n')
#         # f.write('module load torchvision/0.2.1-py27\n')
#         #
#         # for optim_val in optimizers:
#         #     for need_val in need:
#         #         f.write('\n \n')
#         #         H2 = job_name1 + '-' + optim_val.upper() + '-' + str(need_val)
#         #         f.write('python ' + config.main_file + '  --dataset ' + dataset_val + '  --augment ' + str(augment_val)
#         #                 + '  --embedding_size ' + str(ES_val) + '  --batch_size ' + str(batch_size_val)
#         #                 + '  --optimizer '  + optim_val + '  --need  ' + str(need_val) + '  --margin ' + str(margin_val)
#         #                 + ' > ' + folder + os.sep + H2 + '.log')#job_name1
#
# if config.main_file.lower() == 'main_ang.py':
#  batch_sizes = {'cifar100': [200], 'cifar100_20': [40], 'cifar10': [20], 'stl10-reduced': [20]}
#  margins     = {2:[0]}#3:[30, 45] } 1:[30, 45],
#  for setting in margins.keys():
#     margin = margins[setting]
#     # if setting == 1:
#     for dataset in batch_sizes.keys():
#         print config.main_file.lower(), dataset, setting
#         if dataset in ['cifar10', 'cifar100_20', 'cifar100']:
#             batch_size       = batch_sizes[dataset]
#             parameters_list  = [[dataset], ES, batch_size, margin, augment, optimizers]#, need]
#             all_combinations = list(itertools.product(*parameters_list))
#             for parameter_set in all_combinations:
#                 exp_id += 1
#                 dataset_val    = parameter_set[0]
#                 ES_val         = parameter_set[1]
#                 batch_size_val = parameter_set[2]
#                 margin_val     = parameter_set[3]
#                 augment_val    = parameter_set[4]
#                 optim_val      = parameter_set[5]
#                 # need_val       = parameter_set[6]
#                 job_name1      = files[config.main_file] + '-SET-' + str(setting) + '-' + dataset_val + '-ES-' + str(ES_val) \
#                                  + '-BS-' + str(batch_size_val) + '-Margin-' + str(margin_val) + '-AUG-' + str(augment_val) \
#                                  + '-' + optim_val# + '-' + str(need_val)
#                 f = open(job_name1 + '.q', 'w')
#                 f.write('#!/bin/bash\n')
#                 f.write('#SBATCH --job-name=' + job_name1 + '\n')
#                 f.write('#SBATCH --time=04:00:00\n')
#                 f.write('#SBATCH --gres=gpu:1\n')
#                 f.write('#SBATCH --error=err_' + job_name1 + '.txt\n')
#                 f.write('#SBATCH --output=out_' + job_name1 + '.txt\n')
#                 f.write('#SBATCH --ntasks=4\n')
#                 # f.write('#SBATCH --ntasks-per-node=1\n')
#                 f.write('#SBATCH --mem=12g\n')
#                 f.write('#SBATCH --mail-type=ALL\n')
#                 f.write('#SBATCH --mail-user=soumava.roy91@gmail.com\n')
#                 f.write('\n \n')
#                 f.write('module load parallel\n')
#                 f.write('module load pytorch/0.4.0-py27-cuda91\n')
#                 f.write('module load torchvision/0.2.1-py27\n')
#                 H2 = job_name1
#                 f.write('\n \n')
#                 f.write(' seq 2 | parallel -j4 \'python ' + config.main_file + '  --dataset ' + dataset_val + '  --augment ' + str(augment_val)
#                         + '  --embedding_size ' + str(ES_val) + '  --batch_size ' + str(batch_size_val)
#                         + '  --optimizer '  + optim_val  +  ' --setting ' + str(setting) + '  --margin ' + str(margin_val)
#                         + '  --need  {}  > ' + folder + os.sep + H2 + '-"{}".log\'')
#                         # + '  --need  ' + str(need_val)  + ' > ' + folder + os.sep + H2 + '.log')#job_name1
#
#         # if dataset in ['stl10-reduced']:
#         #     batch_size       = batch_sizes[dataset]
#         #     parameters_list  = [[dataset], ES, batch_size, margin, augment]  # , optimizers, need]
#         #     all_combinations = list(itertools.product(*parameters_list))
#         #     for parameter_set in all_combinations:
#         #         exp_id         += 1
#                 # dataset_val     = parameter_set[0]
#                 # ES_val          = parameter_set[1]
#                 # batch_size_val  = parameter_set[2]
#                 # margin_val      = parameter_set[3]
#                 # augment_val     = parameter_set[4]
#                 # optim_val       = parameter_set[5]
#                 # need_val        = parameter_set[6]
#                 # job_name1       = files[config.main_file] + '-SET-' + str(setting) + '-' + dataset_val + '-ES-' + str(ES_val) \
#                 #                   + '-BS-' + str(batch_size_val) + '-Margin-' + str(margin_val) + '-AUG-' + str(augment_val)
#                 #
#                 # f = open(job_name1 + '.q', 'w')
#                 # f.write('#!/bin/bash\n')
#                 # f.write('#SBATCH --job-name=' + job_name1 + '\n')
#                 # f.write('#SBATCH --time=04:00:00\n')
#                 # f.write('#SBATCH --gres=gpu:1\n')
#                 # f.write('#SBATCH --error=err_' + job_name1 + '.txt\n')
#                 # f.write('#SBATCH --output=out_' + job_name1 + '.txt\n')
#                 # f.write('#SBATCH --ntasks-per-node=1\n')
#                 # f.write('#SBATCH --mem=12g\n')
#                 # f.write('#SBATCH --mail-type=ALL\n')
#                 # f.write('#SBATCH --mail-user=soumava.roy91@gmail.com\n')
#                 # f.write('\n \n')
#                 # f.write('module load python\n')
#                 # f.write('module load pytorch/0.4.0-py27-cuda91\n')
#                 # f.write('module load torchvision/0.2.1-py27\n')
#                 # for optim_val in optimizers:
#                 #     for need_val in need:
#                 #         f.write('\n \n')
#                 #         H2 = job_name1 + '-' + optim_val.upper() + '-' + str(need_val)
#                 #         f.write('python ' + config.main_file + '  --dataset ' + dataset_val + '  --augment ' + str(augment_val)
#                 #                 + '  --embedding_size ' + str(ES_val) + '  --batch_size ' + str(batch_size_val)
#                 #                 + '  --optimizer ' + optim_val + '  --need  ' + str(need_val) + ' --setting ' + str(setting)
#                 #                 + '  --margin ' + str(margin_val) + ' > ' + folder + os.sep + H2 + '.log')  # job_name1
# print exp_id
#
