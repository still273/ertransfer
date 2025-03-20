from torch.utils.data import Dataset

class PandasDataset(Dataset):
    def __init__(self, df, attr_column, label_column):
        self.data = df
        self.attr_column = attr_column
        self.label_column = label_column

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        attr = self.data[self.attr_column].iloc[idx]
        label = self.data[self.label_column].iloc[idx]
        return attr, label
