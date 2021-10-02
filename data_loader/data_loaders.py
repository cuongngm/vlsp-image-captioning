import os
import cv2
import json
import string
from random import choice
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from data_loader.base_data_loader import BaseDataLoader
from util.label_convert import LabelConvert
from vncorenlp import VnCoreNLP
ImageFile.LOAD_TRUNCATED_IMAGES = True
convert = LabelConvert(vocab_file='util/vocab_vlsp.txt')


def get_base_info(img_dir, anns_path):
    img_list = []
    label_list = []
    with open(anns_path, 'r') as f:
        all_data = json.load(f)
        for data in all_data:
            img_id = data['id']
            labels = data['captions'].split('\n')
            img_path = os.path.join(img_dir, img_id)
            # label = choice(labels)
            for label in labels:
                img_list.append(img_path)
                label_list.append(label)
    return img_list, label_list


class CaptioningDataset(Dataset):
    def __init__(self, img_dir, anns_path):
        img_list, label_list = get_base_info(img_dir, anns_path)
        assert len(img_list) == len(label_list)
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.label_list[idx]
        words = label.split(' ')
        cap_len = len(words) + 2
        img = Image.open(img_path).resize((256, 256))
        img = img.convert('RGB')
        img = self.transform(img)
        # img.sub_(0.5).div_(0.5)
        return img, label, cap_len


class BertDataset(Dataset):
    def __init__(self, img_dir, anns_path):
        img_list, label_list = get_base_info(img_dir, anns_path)
        assert len(img_list) == len(label_list)
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.table = str.maketrans(dict.fromkeys(string.punctuation))
        self.rdrsegmenter = VnCoreNLP("tachtu/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

    def __len__(self):
        return len(self.img_list)

    def tokenize(self, text):
        try:
            sents = self.rdrsegmenter.tokenize(text)
            text_token = ' '.join([' '.join(sent) for sent in sents])
        except:
            print(text)
            text_token = ''
            print('fail')
        return text_token

    def __getitem__(self, idx):
        info = dict()
        img_path = self.img_list[idx]
        label = self.label_list[idx]
        words = label.split(' ')
        cap_len = len(words) + 2
        label = label.lower()
        label = label.translate(self.table)
        label = self.tokenize(label)
        img = Image.open(img_path).resize((256, 256))
        img = img.convert('RGB')
        img = self.transform(img)
        info['img'] = img
        info['label'] = label
        info['len'] = cap_len
        # img.sub_(0.5).div_(0.5)
        return info


class CaptioningDataLoader(BaseDataLoader):
    """
    captioning data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, anns_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        # self.dataset = CaptioningDataset(img_dir=data_dir, anns_path=anns_path)
        self.dataset = BertDataset(img_dir=data_dir, anns_path=anns_path)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)