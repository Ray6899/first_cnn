import json
import os
import numpy as np

class Read(object):
    def __init__(self, train_sz=0.8):
        self.map_nm_lb = {}
    
    def nm_lb(self, path):
        for x in os.listdir(path):
            self.map_nm_lb[self.map_nm_lb.__len__()] = x

    def _get_key(self, value):
        for k, v in self.map_nm_lb.items():
            if v == value:
                return k

    def get_train(self, path):
        train_data = self._get_name_label(path)
        return train_data
    
    def get_test(self, path):
        test_data = self._get_name_label(path)
        return test_data
    
    def _get_name_label(self, path):
        data = []
        label = []
        for x in os.listdir(path):
            pathx = os.path.join(path, x)
            img_nm_list = os.listdir(pathx)
            for img_nm in img_nm_list:
                img = os.path.join(pathx, img_nm)
                data.append(img)
                label.append(self._get_key(x))
        assert len(data) == len(label)
        return (data, label)
    
    def save_nm_map(self):
        with open('data/split_data/nm_map.json','w+') as f:
            json.dump(self.map_nm_lb,f)
            f.write('\n')

if __name__ == "__main__":
    test_path = os.path.join('E:\\asus\\Python\\my_nn\\data\\split_data','test')
    train_path = os.path.join('E:\\asus\\Python\\my_nn\\data\\split_data','trainX5equal')
    rd = Read()
    rd.nm_lb(test_path)
    train_data = rd.get_train(train_path)
    rd.save_nm_map()
