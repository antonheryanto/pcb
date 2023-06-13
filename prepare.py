import cv2
import torch
from torch.utils.data import Dataset
from os import path
import numpy as np

class fcbData(Dataset):
    def __init__(self, df: pd.DataFrame, IMG_DIR: str, transforms = None):
        self.df = df
        self.img_dir = IMG_DIR
        self.image_ids = self.df['file'].unique()
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ids = image_id.split('_')[1:-1]
        ids[0] = ids[0].title()
        a = '_'.join(ids)
        image_file = path.join(self.img_dir, a, image_id +".jpg")
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        image_values = self.df[self.df['file'] == image_id]
        boxes = image_values[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        class_name = image_values["class"].values[0]
        labels = classes_la[class_name]
        labels = torch.tensor(labels)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.zeros(len(classes_la), dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
        
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return torch.tensor(image), target, image_id

fcb_dataset = fcbData(data, )
image, target, image_id = fcb_dataset[0]
print(target['image_id'], image_id)
fcb_dataset.image_ids