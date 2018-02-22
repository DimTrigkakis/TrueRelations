import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import PIL
from PIL import Image
from torchvision.utils import save_image as save
import random
from models import *
######################### Data Building

class DataBuilder(data.Dataset):

        def __init__(self, configuration):

                self.configuration = configuration
                self.loader = torch.utils.data.DataLoader(dataset=self, batch_size=configuration['sampler']['bs'], shuffle=configuration['sampler']['shuffle'], num_workers=12)

        def __getitem__(self, index):
                datum = self.configuration['data'][index]
                return self.configuration['decoder'](datum)

        def __len__(self):
                return len(self.configuration['data'])

configuration = {'data':[],'decoder':None, 'length':-1, 'sampler':None}
datapath = "/scratch/Jack/datasets/PIPA-relations/train_test_val_splits/annotator_consistency3(used in our paper)/pairwise_face_train_16.txt"

def pair_transform():
    normalize = transforms.Normalize(
        mean=[0.47727655, 0.36052684, 0.30464375],
        std=[0.36995165, 0.3163851, 0.30528845]
    )

    normal_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224, padding=0),
        transforms.ToTensor(), # C x H x W
        normalize
    ])

    return normal_transform

pt = pair_transform()

with open(datapath, "r") as f:
    for line in f.readlines():

        start = str(line).find("faces/")
        line_n = line[start:-1]
        image, label = line_n.split(" ")
        image = image.split("/")[1].split(".jpg")[0]
        image_fA = image[0:2]
        image_fB = image[3:5]
        image_name = image.split("_")[2]+"_"+image.split("_")[3]

        datum = {"face_1": image_fA, "face_2": image_fB, "image": image_name, "relationship": label}

        configuration['data'].append(datum)

def mapping_decoder(datum):

    fA = datum["face_1"]
    fB = datum["face_2"]
    img = datum["image"]
    rel = datum["relationship"]
    mypath = "/scratch/Jack/datasets/Single_Faces_One/Single_Faces/"
    image_file_A = mypath + fA + "_" + img + ".jpg"
    image_file_B = mypath + fB + "_" + img + ".jpg"
    image_A = pt(PIL.Image.open(image_file_A))
    image_B = pt(PIL.Image.open(image_file_B))
    sample = {"ImageA": image_A, "ImageB": image_B, "Label": int(rel)}

    return sample

configuration['subset_percent'] = 0.01 # ~100

def random_subset_selection():
    index = 0
    count_max = len(configuration['data'])
    for i in range(count_max):
        if random.random() >= configuration['subset_percent']:
            configuration['data'].pop(index)
        else:
            index += 1


def specific_subset_selection():
    index = 0
    count_max = len(configuration['data'])
    for i in range(count_max):
        if i % int(1.0/configuration['subset_percent']) != 0:
            configuration['data'].pop(index)
        else:
            index += 1

specific_subset_selection()
configuration['decoder'] = mapping_decoder
configuration['sampler'] = {'bs':32,'shuffle':True}

####

db_train = DataBuilder(configuration)
print(len(db_train))

####

######################### Example Save

#save(datum['ImageA'], "/scratch/Jack/projects/TrueRelations/Vis/ImageA" + str(i) + ".jpg", nrow=1, normalize=False)

####

'''
v = VAE_Adventures(model_version='GammaVersion',\
                   model_choice='FaceConvVae', model_type='Convolutional',\
                   resources=db_train.loader, result_path = "/scratch/Jack/research lab/TrueRelations/Face Pairs/")
'''

v_mix = VAE_Adventures(model_version='DeltaVersion',\
                   model_choice='FaceConvVaeMixture', model_type='Convolutional',\
                   resources=db_train.loader, result_path = "/scratch/Jack/research lab/TrueRelations/Face Pairs/")

######################### Training
v_mix.train()

#for i, datum in enumerate(db_train.loader):
#    break
