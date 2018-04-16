from celeb_gan_builder import *

# Visdom for Visualizations

vis = visdom.Visdom()

######################### Data Building

class Base_Transform():
    def __init__(self, configuration=None):
        self.configuration = configuration

    def t_transform(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.configuration['mean'], std=self.configuration['std'])])


    def ut_transform(self):

        mean, std = self.configuration['mean'], self.configuration['std']
        umean, ustd = [mean[0], mean[1], mean[2]], [std[0], std[1], std[2]]

        for i in range(3):
            ustd[i] = 1.0/ustd[i]
            umean[i] = -umean[i]*ustd[i]

        t = transforms.Compose([transforms.Normalize(mean=umean, std=ustd)])

        return t

class DataBuilder(data.Dataset):

        def __init__(self, configuration):

                self.configuration = configuration
                if "shuffle" in configuration['sampler'].keys():
                    self.loader = data.DataLoader(dataset=self, batch_size=configuration['sampler']['bs'], shuffle=configuration['sampler']['shuffle'], num_workers=12)
                else:
                    self.loader = data.DataLoader(dataset=self, batch_size=configuration['sampler']['bs'], sampler=configuration['sampler']['sampler'], num_workers=12)

                self.ut = Base_Transform(bt_configuration_face).ut_transform()

        def __getitem__(self, index):
                return self.configuration['decoder'](self.configuration['data'][index], self.configuration['transform'])

        def __len__(self):
                return len(self.configuration['data'])

configurations = {'train': {'data':[], 'decoder':None, 'length':-1, 'sampler':None, 'transform':None}, 'val': {'data':[], 'decoder':None, 'length':-1, 'sampler':None, 'transform':None}
                  ,'test': {'data':[], 'decoder':None, 'length':-1, 'sampler':None, 'transform':None}}

datapath ={'train':"/scratch/Jack/datasets/True_Relations_Dataset/train_test_eval_splits/annotator_consistency3(used in our paper)/pairwise_face_train_16.txt",
'val':"/scratch/Jack/datasets/True_Relations_Dataset/train_test_eval_splits/annotator_consistency3(used in our paper)/pairwise_face_eval_16.txt",
'test':"/scratch/Jack/datasets/True_Relations_Dataset/train_test_eval_splits/annotator_consistency3(used in our paper)/pairwise_face_test_16.txt"}

# No cropping or flipping initially

proper_size = (64,64)
bt_configuration_face={'mean':[0.506 ,0.426,0.383],'std':[0.304 ,0.283 ,0.283]}
bt = [Base_Transform(bt_configuration_face).t_transform()]
configurations['train']['transform'] = bt
configurations['test']['transform'] = bt
configurations['val']['transform'] = bt

##### Sampler Init

weight_per_class = [0.] * 16
classes = [x for x in range(16)]
class_numbers = {'train':{x: 0 for x in classes},'val':{x: 0 for x in classes}, 'test':{x: 0 for x in classes}}

######

for face in glob.glob("/scratch/Jack/datasets/CelebA/img_resized_celeba/celebA/*.jpg"):
    datum = {"face": face}
    configurations['train']['data'].append(datum)

# Decode from filepaths to data
def mapping_decoder(datum, t):

    image_file_face = datum["face"]
    face_img = t[0](PIL.Image.open(image_file_face).convert("RGB"))
    sample = {"Face": face_img}

    return sample

############# Subsets

configurations['train']['subset_percent'] = 1 #
configurations['val']['subset_percent'] = 1 #
configurations['test']['subset_percent'] = 1 #

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

############# Set up configurations for sampling and decoding

configurations['train']['decoder'] = mapping_decoder
configurations['train']['sampler'] = {'bs':128,'shuffle':True}
configurations['val']['decoder'] = mapping_decoder
configurations['val']['sampler'] = {'bs':128,'shuffle':False}
configurations['test']['decoder'] = mapping_decoder
configurations['test']['sampler'] = {'bs':128,'shuffle':False}

####

db_train = DataBuilder(configurations['train'])
db_test = DataBuilder(configurations['test'])
db_val = DataBuilder(configurations['val'])

######################################################## MSTD calculation (once)

######### MEAN CALCULATION
'''
face_mean = torch.zeros(3)
face_samples = 0

for i, datum in enumerate(db_train.loader):

    face = datum["Face"]
    face_samples += 1

    face_tensor = face
    for dim in [0,1,1]:
        face_tensor = torch.mean(face_tensor, dim)

    face_mean += face_tensor

    print(face_mean/face_samples)

# bt_configuration_face={'mean':[.4669 ,.3633,.3117],'std':[1,1,1],'transform_list':[transforms.Resize((128,128))]}
# bt_configuration_body={'mean':[0.4410,0.3603,0.3198],'std':[1,1,1],'transform_list':[transforms.Resize((256,128))]}
# bt_configuration_whole={'mean':[0.4402,0.3934,0.3573],'std':[1,1,1],'transform_list':[transforms.Resize((256,256))]}
############## STD CALCULATION

samples = 0
face_var = np.zeros(3)

for i, datum in enumerate(db_train.loader):

    samples += 1
    face = datum["Face"]
    face_var += (np.var(face.numpy(), axis=(0,2,3)))
    print(np.sqrt(face_var/samples))

# print(np.sqrt(face_var/samples), np.sqrt(body_var/samples), np.sqrt(whole_var/samples))

'''
########## VAEDCGAN recombinations

V = GAN_Building(model_choice="DCGAN", dbs={'train':db_train, 'val':db_val, 'test':db_test}, result_path="/scratch/Jack/research lab/True_Relations/", proper_size=proper_size)
V.train()