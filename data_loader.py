# データモジュールを管理する
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# データセットの作成
class DatasetGenerator(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """データセットの行を取得する。

        Args:
            index (list): ほしいデータを指定 [a行:b行, c列:d列]
        """
        data_row = self.data.iloc[index]
        text = data_row['text']
        labels = data_row['label']

        encoding = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels)
        )

# データローダの作成
class DataModuleGenerator(pl.LightningDataModule):
    """
    DataFrameからDataModuleを作成
    """
    def __init__(self, train_df, val_df, test_df, tokenizer, max_length=256, batch_size={'train':32, 'val':256, 'test':256}):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_batch_size = batch_size['train']
        self.val_batch_size = batch_size['val']
        self.test_batch_size = batch_size['test']
        
    def setup(self, stage=None):
        self.train_dataset = DatasetGenerator(self.train_df, self.tokenizer, self.max_length)
        self.val_dataset = DatasetGenerator(self.val_df, self.tokenizer, self.max_length)
        self.test_dataset = DatasetGenerator(self.test_df, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)