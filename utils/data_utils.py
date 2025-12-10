import torch.utils.data as data
import json
import os
import numpy as np
import torch

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

class MyDataset(data.Dataset):
    def __init__(self,
                 data_json_path,
                 npz_folder_path =''):
        self.video_data = load_data(data_json_path)
        self.npz_folder_path = npz_folder_path

    def __getitem__(self, index):
        return self.get_data(index)

    def get_data(self, index):
        npz_path = os.path.join(self.npz_folder_path, f"{self.video_data[index]['video']}.npz")
        npz_file = np.load(npz_path)
        video_feature = npz_file['video_feature']
        text_feature = npz_file['text_feature']
        label = npz_file['label']
        return video_feature, text_feature, label

    def __len__(self):
        return len(self.video_data)
    
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data = MyDataset(os.path.join(current_dir, "datasets/train.json"), os.path.join(current_dir, "data"))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=False, num_workers=16, pin_memory=True)
    for batch in train_loader:
        video_features, text_features, labels = batch
        print(video_features.size())
        print(text_features.size())
        print(labels.size())
        break