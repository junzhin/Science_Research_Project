import argparse
from email.policy import default
import os
import random
import shutil
from termios import PARENB
import time
import warnings
import pandas as pd
import random
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


datafile = r'/data/gpfs/datasets/Imagenet'
training_filename = r'/train_blurred'
valid_filename = r'/val_blurred'

traindir = datafile + training_filename
valdir =  datafile + valid_filename
####################################################
print("--"*20)
print("training directory:" + traindir + "\n")
print("validating directory:" + valdir + "\n")
print("--"*20)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_dataset_initial = datasets.ImageFolder(traindir, 
transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
print(train_dataset_initial)
print(train_dataset_initial.class_to_idx)
print("--"*20)

######################################################################################################################
# extend the functionality to include the subset model training_filename
valid_dataset_initial = datasets.ImageFolder(valdir, 
transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

print(valid_dataset_initial)
print("--"*20)
randomchosenmode = False

masked_label_names = list(pd.read_csv(r'/home/junzhin/Project/Summer_project/code/version1.0/masked_office31_imagenetlabel2_manual_df.csv',header = None)[0])
print(masked_label_names)
non_masked_label_indices = [train_dataset_initial.class_to_idx[x] for x in train_dataset_initial.classes if x not in masked_label_names]
# non_masked_label_names = [x for x in train_dataset_initial.classes if x not in masked_label_names]
chosen_idxs = [indx for indx, target_idx in enumerate(train_dataset_initial.targets) if target_idx in non_masked_label_indices]
print(len(chosen_idxs))







#     masked_label_list= pd.read_csv(subsetpath, header = None)
#     print(masked_label_list)
#     if randomchosenmode:
#         # fix the random seed to produce the replicable sampling results
#         random.seed(1000)

#         # select the classes after excluding the masked classes 
#         exclude_masked_classes = [one_class for one_class in train_dataset_initial.classes if one_class not in list(masked_label_list[0])]

#         # assign the number of classes to args 
#         num_classes = len(exclude_masked_classes)
#         print("+---"*20)
#         print(len(exclude_masked_classes))
#         print(len([train_dataset_initial.class_to_idx[c] for c in list(masked_label_list[0])]))
#         print("+---"*20)

#         random_selected_classes = random.sample(exclude_masked_classes, len(masked_label_list))
#         chosen_classes_labels_indices = [train_dataset_initial.class_to_idx[each] for each in train_dataset_initial.classes if each not in random_selected_classes]
#         print(len(chosen_classes_labels_indices))

#         classes_dict = {x:i for i,x in enumerate(chosen_classes_labels_indices)}

#         chosen_classes_labels_names = [each for each in train_dataset_initial.classes if each not in random_selected_classes]
#         # print(chosen_classes_labels_names)

#         # Save the random selected_classes to a csv file
#         random_selected_classes_labels_names_df = pd.DataFrame(random_selected_classes)
#         print(random_selected_classes_labels_names_df)
#         random_selected_classes_labels_names_df.to_csv('./random_selected_classes_labels.csv', index = False, header = False)

#         # Find all relevant indices in the training and validating sets
#         chosen_index_train = [index for index in range(len(train_dataset_initial)) if train_dataset_initial.imgs[index][1] in chosen_classes_labels_indices]
#         chosen_index_valid = [index for index in range(len(valid_dataset_initial)) if valid_dataset_initial.imgs[index][1] in chosen_classes_labels_indices]
#     else:
#         # assigned the number of classes to args 
#         masked_classes = [train_dataset_initial.class_to_idx[c] for c in list(masked_label_list[0])]
#         num_classes = len(train_dataset_initial.classes) - len(masked_classes)
#         print("+---"*20)
#         print(num_classes)
#         print(len(masked_classes))
#         print("+---"*20)
        
#         chosen_index_train = [index for index in range(len(train_dataset_initial)) if train_dataset_initial.imgs[index][1] not in masked_classes]
#         chosen_index_valid = [index for index in range(len(valid_dataset_initial)) if valid_dataset_initial.imgs[index][1] not in masked_classes]

#         chosen_classes_labels_indices = [train_dataset_initial.class_to_idx[each] for each in train_dataset_initial.classes if each not in list(masked_label_list[0])]
#         classes_dict = {x:i for i,x in enumerate(chosen_classes_labels_indices)}

#     train_dataset = torch.utils.data.Subset(train_dataset_initial, chosen_index_train)
#     valid_dataset = torch.utils.data.Subset(valid_dataset_initial, chosen_index_valid)
#     print(len(chosen_index_train))
#     print("train_datast length is %d" % (len(train_dataset)))
#     print(len(chosen_index_valid))
#     print("valid_dataset length is %d" % (len(valid_dataset)))
#     warnings.warn('Since you do not specify the csv file for the class labels you are going to mask, so no subset model training will be used in this case!')

# train_dataset, valid_dataset = train_dataset_initial,valid_dataset_initial
# classes_dict = {i:i for i in range(len(train_dataset.classes))}

# print(classes_dict)