import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class BasicDataset(Dataset):

    def __init__(self,patch_h,patch_w,datasetName,netType,train_mode=False):

        self.patch_h = patch_h
        self.patch_w = patch_w

        if netType == 'unet' or netType == 'deeplabv3plus':
            self.imgTrans = False
        else: 
            self.imgTrans = True

        self.transform = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
        ])

        self.dataset = datasetName

        if datasetName == 'core_seg':
            self.n1 = 1250
            self.n2 = 1250

            # self.train_data_dir = '../raw_datasets/res_11/res_p_valid_input'
            # self.train_label_dir = '../raw_datasets/res_11/res_p_valid_label'
            # self.valid_data_dir = '../raw_datasets/res_11/res_p_input'
            # self.valid_label_dir = '../raw_datasets/res_11/res_p_label'


            #run demo_
            self.valid_data_dir = '../raw_datasets/input_light'
            self.valid_label_dir = '../raw_datasets/label_light'
            self.train_data_dir = '../raw_datasets/input_light'
            self.train_label_dir = '../raw_datasets/label_light'
        elif datasetName == 'core_seg_p2p':
            self.n1 = 1250
            self.n2 = 1250

            # #train demo_
            # self.train_data_dir = '../datasets_raw/radon_process_d-l/p_input'
            # self.train_label_dir = '../datasets_raw/radon_process_d-l/p_label'
            # self.valid_data_dir = '../datasets_raw/radon_process_d-l/p_valid_input'
            # self.valid_label_dir = '../datasets_raw/radon_process_d-l/p_valid_label'
            # valid
            self.valid_data_dir = '../datasets_raw/radon_process_d-l/p_input'
            self.valid_label_dir = '../datasets_raw/radon_process_d-l/p_label'
            self.train_data_dir = '../datasets_raw/radon_process_d-l/p_valid_input'
            self.train_label_dir = '../datasets_raw/radon_process_d-l/p_valid_label'

        elif datasetName == 'core_seg_o2p':
            self.n1 = 1250
            self.n2 = 1250

            self.train_data_dir = '../datasets_raw/radon_process_d-l/ori_input'
            self.train_label_dir = '../datasets_raw/radon_process_d-l/p_label'
            self.valid_data_dir = '../datasets_raw/radon_process_d-l/ori_valid_input'
            self.valid_label_dir = '../datasets_raw/radon_process_d-l/p_valid_label'



        else datasetName == 'core_seg_model':
            self.n1 = 256
            self.n2 = 256
            self.train_data_dir = '../raw_datasets/model_artifact_data/ring_impact/res_p_input'
            self.train_label_dir = '../raw_datasets/model_artifact_data/ring_impact/res_p_label'

            self.valid_data_dir = '../raw_datasets/model_artifact_data/ring_impact/res_p_valid_input'
            self.valid_label_dir = '../raw_datasets/model_artifact_data/ring_impact/res_p_valid_label'



            # self.valid_data_dir = '../raw_datasets/model_artifact_data/res_p_valid_input'
            # self.valid_label_dir = '../raw_datasets/model_artifact_data/res_p_valid_label'
            # self.train_data_dir = '../raw_datasets/model_artifact_data/res_p_input'
            # self.train_label_dir = '../raw_datasets/model_artifact_data/res_p_label'

            # #run demo_
            # self.valid_data_dir = './raw_datasets/model_artifact_data/res_p_input'
            # self.valid_label_dir = './raw_datasets/model_artifact_data/res_p_label'
            # self.train_data_dir = './raw_datasets/model_artifact_data/res_p_valid_input'
            # self.train_label_dir = './raw_datasets/model_artifact_data/res_p_valid_label'


        
        else:
            print("Dataset error!!")
        print('netType:' + netType)
        print('dataset:' + datasetName)
        print('patch_h:' + str(patch_h))
        print('patch_w:' + str(patch_w))

        if train_mode:
            self.data_dir = self.train_data_dir
            self.label_dir = self.train_label_dir

            self.ids = len(os.listdir(self.label_dir)) // 2
            print(self.ids)
        else:
            self.data_dir = self.valid_data_dir
            self.label_dir = self.valid_label_dir

            self.ids = len(os.listdir(self.label_dir)) // 2
            print(self.ids)


    def __len__(self):
        return self.ids

    def __getitem__(self,index):
        dPath = os.path.join(self.data_dir, f'slice_{index:04d}.npy')
    	tPath = os.path.join(self.label_dir, f'slice_{index:04d}.npy')
    	print(tPath)
    	# dPath = os.path.join(self.data_dir, f'{index}.npy')
    	# tPath = os.path.join(self.label_dir, f'{index}.npy')

    	#
    	# dPath = self.data_dir + '/slice_00' + str(index) + '.npy'
    	# tPath = self.label_dir + '/' + str(index) + '.npy'
    	# print(dPath, tPath)

    	data = np.load(dPath).reshape(self.n1, self.n2)
    	label = np.load(tPath).reshape(self.n1, self.n2)  # lazy to modify; label unify with 0,1,2 ;
    	# data = np.load(dPath)
    	# data = np.load(dPath)
    	# label = np.load(tPath)      
        # import pdb;pdb.set_trace()
        data = np.reshape(data,(1,1,self.n1,self.n2))
        data = np.concatenate([data,self.data_aug(data)],axis=0)
        label = np.reshape(label,(1,1,self.n1,self.n2))
        label = np.concatenate([label,self.data_aug(label)],axis=0)
        label = label-1  # lazy to modify
        if self.imgTrans:
            img_tensor = np.zeros([2,1,self.patch_h*14,self.patch_w*14],np.float32)
            for i in range(data.shape[0]):
                img = Image.fromarray(np.uint8(data[i,0]))
                img_tensor[i,0] = self.transform(img)
            # print(data.shape)
            data = img_tensor
            data = data.repeat(3,axis=1)
        elif not self.imgTrans:
            data = data/255


        return data,label

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.fliplr(np.squeeze(data))
        return data_fliplr.reshape(b,c,h,w)


if __name__ == '__main__':
    train_set = BasicDataset(64,64,'core_seg','unet',False)
    print(train_set.__getitem__(0)[1].shape)
    print(len(train_set))

'''
2025-07-16 12:24:31,401 - INFO - Class 0: 98.38%
2025-07-16 12:24:31,402 - INFO - Class 1: 93.11%
2025-07-16 12:24:31,402 - INFO - Class 2: 35.32%
2025-07-16 12:24:31,402 - INFO - miou: 0.611422
2025-07-16 12:24:31,402 - INFO - F1: 0.503761
2025-07-16 12:24:31,402 - INFO - mpa: 0.797487
'''
