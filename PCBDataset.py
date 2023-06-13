import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import xml.etree.ElementTree as ET
from os import walk, path

class PCBDataset(Dataset):
    def __init__(self, file: str = None):
        if file:
            self.images, self.labels = torch.load(file)

    def init(self, root: str, transforms: transforms.Compose = None, sequence_length = 6):
        self.image_path = path.join(root, "images")
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.df = self.parse(path.join(root, "Annotations"))
        classes = self.df["class"].unique()
        self.classes_la = {'pass': 0}
        for i in range(len(classes)):
            self.classes_la[classes[i]] = i + 1
        groups = self.df["file"].unique()
        # df_grp = self.df.groupby(['file'])
        self.images = []
        self.labels=[]
        for group in groups:
            row = self.df[self.df['file'] == group]
            image, label = self.get_item(row)
            self.images.append(image)
            self.labels.append(label)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        lc = 0
        for l in label:
            if l != 0:
                lc += 1
        return img, label, torch.tensor(lc)

    def __len__(self):
        return len(self.labels)

    def get_image_path(self, image_id: str):
        ids = image_id.split('_')[1:-1]
        ids[0] = ids[0].title()
        class_path = '_'.join(ids)
        image_file = path.join(self.image_path, class_path, image_id +".jpg")
        return image_file

    def get_item(self, image_values: pd.DataFrame):
        pd_class = image_values["class"]
        class_id = self.classes_la[pd_class.values[0]]
        labels = torch.zeros(self.sequence_length, dtype=torch.long)
        for i in range(pd_class.count()):
            labels[i] = class_id
        image_id = image_values['file'].values[0]
        ids = image_id.split('_')[1:-1]
        ids[0] = ids[0].title()
        class_path = '_'.join(ids)
        image_file = path.join(self.image_path, class_path, image_id +".jpg")
        image = Image.open(image_file)
        if not self.transforms:
            self.transforms = transforms.Compose([ transforms.ToTensor() ])
        image = self.transforms(image)
        return image, labels

    def parse(self, dataset_path):
        dataset = {
            "xmin":[],
            "ymin":[],   
            "xmax":[],
            "ymax":[],
            "class":[],    
            "file":[],
            "width":[],
            "height":[],
        }
        all_files = []
        for root, subdirs, files in walk(dataset_path):
            # print([root, subdirs, files])
            for name in files:
                anno = path.join(root, name)                
                self.parseData(anno, dataset)
        return pd.DataFrame(dataset)
        
    def parseData(self, anno: str, dataset: dict[str, list]):
        filename,_ = path.splitext(path.split(anno)[1])
        tree = ET.parse(anno)
        for elem in tree.iter():
            if 'size' in elem.tag:
                # print('[size] in elem.tag ==> list(elem)\n'), print(list(elem))
                for attr in list(elem):
                    if 'width' in attr.tag: 
                        width = int(round(float(attr.text)))
                    if 'height' in attr.tag:
                        height = int(round(float(attr.text)))

            if 'object' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        name = attr.text                 
                        dataset['class']+=[name]
                        dataset['width']+=[width]
                        dataset['height']+=[height] 
                        dataset['file']+=[filename]
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                dataset['xmin']+=[xmin]
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                dataset['ymin']+=[ymin]                                
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                dataset['xmax']+=[xmax]                                
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                dataset['ymax']+=[ymax]


