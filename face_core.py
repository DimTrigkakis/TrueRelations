import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os.path as path
import PIL
from PIL import Image
import random
import visdom
import numpy as np
from recombined_gan import *
from face_gan_experiment import *

# Visdom for Visualizations

vis = visdom.Visdom()

######################### Data Building

class Base_Transform():
    def __init__(self, configuration=None):
        self.configuration = configuration

    def t_transform(self):

        t = transforms.Compose([
            *self.configuration['transform_list'],
            transforms.ToTensor(), # C x H x W
            transforms.Normalize(mean=self.configuration['mean'], std=self.configuration['std'])
        ])

        return t

    def ut_transform(self):

        mean = self.configuration['mean']
        std = self.configuration['std']
        umean = [mean[0], mean[1], mean[2]]
        ustd = [std[0], std[1], std[2]]
        for i in range(3):
            ustd[i] = 1.0/ustd[i]
            umean[i] = -umean[i]*ustd[i]

        t = transforms.Compose([
            transforms.Normalize(mean=umean, std=ustd),
        ])

        return t

class DataBuilder(data.Dataset):

        def __init__(self, configuration):

                self.configuration = configuration
                if "shuffle" in configuration['sampler'].keys():
                    self.loader = torch.utils.data.DataLoader(dataset=self, batch_size=configuration['sampler']['bs'], shuffle=configuration['sampler']['shuffle'], num_workers=12)
                else:
                    self.loader = torch.utils.data.DataLoader(dataset=self, batch_size=configuration['sampler']['bs'], sampler=configuration['sampler']['sampler'], num_workers=12)

                self.ut = Base_Transform(bt_configuration_face).ut_transform()

        def __getitem__(self, index):
                datum = self.configuration['data'][index]
                datum_random = self.configuration['data'][random.randint(0,len(self.configuration['data'])-1)]
                return {'normal':self.configuration['decoder'](datum, self.configuration['transform']),'random': self.configuration['decoder'](datum_random, self.configuration['transform'])}

        def __len__(self):
                return len(self.configuration['data'])

configurations = {'train': {'data':[], 'decoder':None, 'length':-1, 'sampler':None, 'transform':None}, 'val': {'data':[], 'decoder':None, 'length':-1, 'sampler':None, 'transform':None}
                  ,'test': {'data':[], 'decoder':None, 'length':-1, 'sampler':None, 'transform':None}}

datapath ={'train':"/scratch/Jack/datasets/True_Relations_Dataset/train_test_eval_splits/annotator_consistency3(used in our paper)/pairwise_face_train_16.txt",
'val':"/scratch/Jack/datasets/True_Relations_Dataset/train_test_eval_splits/annotator_consistency3(used in our paper)/pairwise_face_eval_16.txt",
'test':"/scratch/Jack/datasets/True_Relations_Dataset/train_test_eval_splits/annotator_consistency3(used in our paper)/pairwise_face_test_16.txt"}

# No cropping or flipping initially

'''
bt_configuration_face={'mean':[.4669 ,.3633,.3117],'std':[0.26307311 ,0.23352264 ,0.22752409],'transform_list':[transforms.Resize((128,128))]}
bt_configuration_body={'mean':[0.4410,0.3603,0.3198],'std':[0.2756887 ,0.25631477 ,0.25326068] ,'transform_list':[transforms.Resize((256,128))]}
bt_configuration_whole={'mean':[0.4402,0.3934,0.3573],'std':[0.275962 ,0.26490119 ,0.26475916],'transform_list':[transforms.Resize((256,256))]}
'''

bt_configuration_face={'mean':[.4669 ,.3633,.3117],'std':[0.26307311 ,0.23352264 ,0.22752409],'transform_list':[transforms.Resize((224,224))]}
bt_configuration_body={'mean':[0.4410,0.3603,0.3198],'std':[0.2756887 ,0.25631477 ,0.25326068] ,'transform_list':[transforms.Resize((224,224))]}
bt_configuration_whole={'mean':[0.4402,0.3934,0.3573],'std':[0.275962 ,0.26490119 ,0.26475916],'transform_list':[transforms.Resize((224,224))]}

#bt_configuration_body={'mean':[0.4410,0.3603,0.3198],'std':[0.2756887 ,0.25631477 ,0.25326068] ,'transform_list':[transforms.Resize((256,256)), transforms.RandomCrop((224,224))]}
#bt_configuration_whole={'mean':[0.4402,0.3934,0.3573],'std':[0.275962 ,0.26490119 ,0.26475916],'transform_list':[transforms.Resize((256,256)), transforms.RandomCrop((224,224))]}

bt = [Base_Transform(bt_configuration_face).t_transform(), Base_Transform(bt_configuration_body).t_transform(), Base_Transform(bt_configuration_whole).t_transform()]

configurations['train']['transform'] = bt
configurations['test']['transform'] = bt
configurations['val']['transform'] = bt

##### Sampler Init

weight_per_class = [0.] * 16
classes = [x for x in range(16)]
class_numbers = {'train':{x: 0 for x in classes},'val':{x: 0 for x in classes}, 'test':{x: 0 for x in classes}}

######

for subset in ["train", "val", "test"]:
    with open(datapath[subset], "r") as f:
        for line in f.readlines():

            start = str(line).find("faces/")
            line_n = line[start:-1]
            image, label = line_n.split(" ")
            image = image.split("/")[1].split(".jpg")[0]
            image_fA = image[0:2]
            image_fB = image[3:5]
            image_name = image.split("_")[2]+"_"+image.split("_")[3]

            ln = int(label)

            if ln <= 3:
                domain = '0'
            elif ln <= 6:
                domain = '1'
            elif ln == 7:
                domain = '2'
            elif ln <= 11:
                domain = '3'
            else:
                domain = '4'

            datum = {"face_1": image_fA, "face_2": image_fB, "image": image_name, "relationship": label, 'domain': domain}
            class_numbers[subset][int(label)] += 1
            configurations[subset]['data'].append(datum)

# Decode from filepaths to data
def mapping_decoder(datum, t):

    fA = datum["face_1"]
    fB = datum["face_2"]
    img = datum["image"]
    rel = datum["relationship"]
    dom = datum["domain"]
    mypath_face = "/scratch/Jack/datasets/True_Relations_Dataset/all_single_face/"
    mypath_body = "/scratch/Jack/datasets/True_Relations_Dataset/all_single_body/"
    mypath_whole = "/scratch/Jack/datasets/True_Relations_Dataset/PIPA_original/"
    image_file_A_face = mypath_face + fA + "_" + img + ".jpg"
    image_file_B_face = mypath_face + fB + "_" + img + ".jpg"
    image_file_A_body = mypath_body + fA + "_" + img + ".jpg"
    image_file_B_body = mypath_body + fB + "_" + img + ".jpg"
    true_label = -1

    for i, subset_type in enumerate(['leftover','train','val','test']):
        image_file_whole = mypath_whole + subset_type +"/"+img + ".jpg"
        if path.isfile(image_file_whole):
            true_label = i
            break

    iAf = t[0](PIL.Image.open(image_file_A_face))
    iBf = t[0](PIL.Image.open(image_file_B_face))
    iAb = t[1](PIL.Image.open(image_file_A_body))
    iBb = t[1](PIL.Image.open(image_file_B_body))
    iW = t[2](PIL.Image.open(image_file_whole))
    label = torch.LongTensor([int(rel)])
    domain = torch.LongTensor([int(dom)])
    sample = {"FaceA": iAf, "FaceB": iBf, "BodyA":iAb, "BodyB":iBb, "Whole":iW, 'Label': label, 'Domain': domain, "Subset" : true_label}

    return sample

############# Subsets

configurations['train']['subset_percent'] = 1.0 #
configurations['val']['subset_percent'] = 1.0 #
configurations['test']['subset_percent'] = 1.0 #

def random_subset_selection():
    for subset in ['train','val','test']:
        index = 0
        count_max = len(configurations[subset]['data'])
        for i in range(count_max):
            if random.random() >= configurations[subset]['subset_percent']:
                datum = configurations[subset]['data'].pop(index)
                if subset == "train":
                    label = datum["relationship"]
                    class_numbers[subset][int(label)] -= 1
            else:
                index += 1
def specific_subset_selection():
    for subset in ['train','val','test']:
        index = 0
        count_max = len(configurations[subset]['data'])
        for i in range(count_max):
            if i % int(1.0/configurations[subset]['subset_percent']) != 0:
                datum = configurations[subset]['data'].pop(index)
                if subset == "train":
                    label = datum["relationship"]
                    class_numbers[subset][int(label)] -= 1
            else:
                index += 1

specific_subset_selection()

############## Uniform Sampling

train_count = float(len(configurations['train']["data"]))
weight = [0] * int(train_count)
weights = torch.DoubleTensor(weight)
for i in range(16):
    if class_numbers['train'][classes[i]] == 0:
        weight_per_class[i] = 0
        continue

    weight_per_class[i] = train_count/class_numbers['train'][classes[i]]

for idx, item in enumerate(configurations['train']["data"]):
    label = -1
    for i, c in enumerate(classes):
        if c == int(item["relationship"]):
            label = i
    assert label != -1
    weight[idx] = weight_per_class[label]

############# Set up configurations for sampling and decoding

train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

configurations['train']['decoder'] = mapping_decoder
configurations['train']['sampler'] = {'bs':4,'sampler':train_sampler} # train sampler and subset are mutually exclusive
configurations['val']['decoder'] = mapping_decoder
configurations['val']['sampler'] = {'bs':4,'shuffle':False}
configurations['test']['decoder'] = mapping_decoder
configurations['test']['sampler'] = {'bs':4,'shuffle':False}

####

db_train = DataBuilder(configurations['train'])
db_test = DataBuilder(configurations['test'])
db_val = DataBuilder(configurations['val'])

######################################################## MSTD calculation (once)
'''
######### MEAN CALCULATION

face_mean = torch.zeros(3)
face_samples = 0
body_mean = torch.zeros(3)
body_samples = 0
whole_mean = torch.zeros(3)
whole_samples = 0

for i, datum in enumerate(db_train.loader):

    facesA = datum["FaceA"]
    facesB = datum["FaceB"]
    face_samples += 2

    face_tensor = facesA
    for dim in [0,1,1]:
        face_tensor = torch.mean(face_tensor, dim)

    face_mean += face_tensor

    face_tensor = facesB
    for dim in [0,1,1]:
        face_tensor = torch.mean(face_tensor, dim)

    face_mean += face_tensor

    bodiesA = datum["BodyA"]
    bodiesB = datum["BodyB"]
    body_samples += 2

    body_tensor = bodiesA
    for dim in [0,1,1]:
        body_tensor = torch.mean(body_tensor, dim)

    body_mean += face_tensor

    body_tensor = bodiesB
    for dim in [0,1,1]:
        body_tensor = torch.mean(body_tensor, dim)

    body_mean += body_tensor

    whole = datum["Whole"]
    whole_samples += 1

    whole_tensor = whole
    for dim in [0,1,1]:
        whole_tensor = torch.mean(whole_tensor, dim)

    whole_mean += whole_tensor

# print(face_mean/face_samples, body_mean/body_samples, whole_mean/whole_samples)

# bt_configuration_face={'mean':[.4669 ,.3633,.3117],'std':[1,1,1],'transform_list':[transforms.Resize((128,128))]}
# bt_configuration_body={'mean':[0.4410,0.3603,0.3198],'std':[1,1,1],'transform_list':[transforms.Resize((256,128))]}
# bt_configuration_whole={'mean':[0.4402,0.3934,0.3573],'std':[1,1,1],'transform_list':[transforms.Resize((256,256))]}

############## STD CALCULATION

samples = 0
face_var = np.zeros(3)
body_var = np.zeros(3)
whole_var = np.zeros(3)

for i, datum in enumerate(db_train.loader):

    samples += 1
    facesA, facesB = datum["FaceA"], datum["FaceB"]
    face_var += (np.var(facesA.numpy(), axis=(0,2,3))+np.var(facesB.numpy(), axis=(0,2,3)))/2
    bodiesA, bodiesB = datum["BodyA"], datum["BodyB"]
    body_var += (np.var(bodiesA.numpy(), axis=(0,2,3)) + np.var(bodiesB.numpy(), axis=(0,2,3)))/2

    whole = datum["Whole"]
    whole_var += np.var(whole.numpy(), axis=(0,2,3))

# print(np.sqrt(face_var/samples), np.sqrt(body_var/samples), np.sqrt(whole_var/samples))
'''
######################### Visuals
'''
for i, datum in enumerate(db_train.loader):
    vis.image(datum["FaceA"][0].numpy())
    vis.image(datum["FaceB"][0].numpy())
    vis.image(datum["BodyA"][0].numpy())
    vis.image(datum["BodyB"][0].numpy())
    vis.image(datum["Whole"][0].numpy())
    break
'''
######################### Start True Script Here
'''
for subset in ['train','val','test']:
    print(class_numbers[subset])
    c = 0
    m = 0
    for i in range(16):
        c += class_numbers[subset][i]
        if class_numbers[subset][i] > m:
            m = class_numbers[subset][i]

    print(c, m/(1.0*c))
'''
########## VAE multi-cluster

V = GAN_Building(model_choice="GAN Zodiac", dbs={'train':db_train, 'val':db_val, 'test':db_test}, result_path="/scratch/Jack/research lab/True_Relations/")
V.train()

############# BASELINES
'''
from models import StreamResnet as SR
import torch

sr = SR(dbs={'train':db_train, 'val':db_val, 'test':db_test}, config={"streams":2, "epochs":1, 'criterion' : nn.CrossEntropyLoss(),'optimizer': "Adam"})

for aeon in range(15):
    print("aeon : {}".format(aeon+1))
    sr.perform(db_type="train", method="train")
    sr.perform(db_type="train", method="acc")
    sr.perform(db_type="val", method="acc")
    sr.perform(db_type="test", method="acc")

# DEBUG remember eval and train modes
# DEBUG remember inference
# DEBUG remember that ut changes the base image

# Leaderboard:
# Best Test on 2-stream should be 46% (that's the number to beat)
# 4-stream, val 44.64%, test 29.35%
# 2-stream, val 32.87%, test 29.76%

# TO DO Conf Matrix, Per class accuracies, full analysis plots loss etc

# Final Proper 2-stream build, val 32.58%, test 29.21%
'''