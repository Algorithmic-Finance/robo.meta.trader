import torch
import pickle
import math

from sklearn.model_selection import train_test_split
from datetime import datetime
from torch.utils.data import DataLoader


class StockDatagenInMemmory:
    def __init__(self, train_path: str, label_path: str):
        
        self.train_path = train_path
        self.label_path = label_path
        self.X = self.__load_pickled_data(train_path)
        self.y = self.__load_pickled_data(label_path)
    
    def get_data_loaders(self, train_batch_size: int, val_batch_size: int, num_workers: int):
        
        sec_seed = datetime.now().second
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.20, random_state = sec_seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.12, random_state = sec_seed)
        
        train_data_loader = DataGeneratorInMemmory(X_train, y_train, train_batch_size)
        val_data_loader = DataGeneratorInMemmory(X_val, y_val, val_batch_size)
        test_data_loader = DataGeneratorInMemmory(X_test, y_test, val_batch_size)
        
        print(f'Initialized Data Loaders:', f' Train set size: {len(X_train)}', f'Validation set size {len(X_val)}', f'Test set size {len(X_test)}', sep = '\n')
        return DataLoader(train_data_loader, num_workers = num_workers, pin_memory=True,  shuffle = True), \
               DataLoader(val_data_loader, num_workers = num_workers, pin_memory=True), \
               DataLoader(test_data_loader, num_workers = num_workers, pin_memory=True)
    
    def __load_pickled_data(self, path: str):
        
        print(f'Loading pickled data from the file: {path}')
        with open(path,'rb') as pickled_data:
            X = pickle.load(pickled_data)
        
        return X


class DataGeneratorInMemmory(torch.utils.data.Dataset):
    
    def __init__(self, X, y, batch_size: int):

        self.batch_size = batch_size
        self.X = X
        self.y = y
        
    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        from_idx = index * self.batch_size
        to_idx =  len(self.X) if (index + 1) * self.batch_size > len(self.X) else (index + 1) * self.batch_size
        X = self.X[from_idx : to_idx] 
        y = self.y[from_idx : to_idx] 
            
        return torch.mul(torch.tensor(X, dtype = torch.float32),1/255).permute(0,3,1,2), torch.tensor(y, dtype = torch.float32)