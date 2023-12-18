import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    def __init__(self, data, sequence_length):
        
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - len(self.sequence_length)
    
    def __getitem__(self, index):
        start_index = index
        end_index = index + self.sequence_length

        input_sequence = self.data[start_index:end_index]
        target = self.data[end_index+1]

        return input_sequence, target
    
if __name__ == "__main__":
    import pandas as pd
    sample_dataset = pd.read_csv("dataset/temp.csv")
    print(sample_dataset.head())
    dataloader = DataLoader(sample_dataset, batch_size=5)

    print(next(iter(dataloader)))
