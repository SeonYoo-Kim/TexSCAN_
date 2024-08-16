import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASS_NAMES = ['carpet'] # 테스트용
#CLASS_NAMES = ['carpet', 'grid', 'leather', 'tile', 'wood']
#CLASS_NAMES = ['02']
#CLASS_NAMES = ['texture_1', 'texture_2']
#CLASS_NAMES = ['grey_cloth', 'grid_cloth', 'pink_flower', 'yellow_cloth']
#CLASS_NAMES = ['Blotchy_099', 'Fibrous_183', 'Marbled_078', 'Matted_069', 'Mesh_114', 'Perforated_037', 'Stratified_154', 'Woven_001', 'Woven_068', 'Woven_104', 'Woven_125', 'Woven_127']

class MVTecDataset(Dataset):
    def __init__(self, root_path='../data', class_name='carpet', is_train=True, resize=320, cropsize=320):

        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.mvtec_folder_path = os.path.join(root_path, 'MVTec')

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0: # 해당 인덱스가 정상이면
            mask = torch.zeros([1, self.cropsize, self.cropsize]) # 까만 마스크 생성
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask # 라벨(0, 1)과 텐서 변환 & resize된 img, mask 반환

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []
        # x : png이미지 하나하나의 경로가 sorted된 리스트, 모든 타입의 defect 다 있음
        # y : x의 요소 하나하나에 대한 라벨, 정상=0, 이상=1

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, phase)
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list)) # y 리스트에 정상 라벨 (0) 추가
                mask.extend([None] * len(img_fpath_list)) # mask 리스트에 None 추가
            else:
                y.extend([1] * len(img_fpath_list)) # y 리스트에 이상 라벨 (1) 추가
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list) # mask 리스트에 GT mask 경로 추가

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)