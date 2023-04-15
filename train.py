import json
import pandas as pd
import random
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data_loader import DataModuleGenerator
from model import ModelForSequenceClassification_pl

# 事前学習済みモデルの指定 ---------------------
MODEL_NAME = 'ku-nlp/deberta-v2-base-japanese'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
output_dir = 'output/jnli_deberta-base_2'

# データの読み込み
data_dir = 'data/JGLUE/datasets/jnli-v1.1'
train_datalist = [json.loads(line) for line in open(data_dir + '/train-v1.1.json', encoding='utf-8')]
# train_datalistをシャッフルしておく
random.seed(42) # シード値は固定
random.shuffle(train_datalist)
test_datalist = [json.loads(line) for line in open(data_dir + '/valid-v1.1.json', encoding='utf-8')]

train_num = int(len(train_datalist) * 0.9) # 訓練データと検証データに分割する数

print('train_data_num :', train_num)
print('valid_data_num :', len(train_datalist) - train_num)
print('test_data_num  :', len(test_datalist))
    
# データフレーム化
train_df = pd.DataFrame(train_datalist[:train_num])
val_df = pd.DataFrame(train_datalist[train_num:])
test_df = pd.DataFrame(test_datalist)

# Checkpoint
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    save_top_k = 1,
    save_weights_only = True,
    dirpath = output_dir + '/model/'
)

# Early_stopping
early_stopping = EarlyStopping(
    monitor = 'val_loss', 
    verbose = True, 
    mode = "min", 
    patience = 3
)

# Logger
tb_logger = pl_loggers.TensorBoardLogger(
    save_dir = output_dir + "/logs/"
)

trainer = pl.Trainer(
    accelerator = 'gpu', 
    devices = [1], # 1だと個数、[1]だと番号指定
    max_epochs = 10,
    callbacks = [checkpoint, early_stopping],
    logger = tb_logger
)

# DataFrameを作成し、setupする
data_module = DataModuleGenerator(
    train_df = train_df,
    val_df = val_df, 
    test_df = test_df, 
    tokenizer = tokenizer, 
    max_length = 128,
    batch_size = {'train':32, 'val':256, 'test':256}
)
data_module.setup()

# モデル定義
model = ModelForSequenceClassification_pl(
    MODEL_NAME, 
    num_labels = 3, 
    lr = 1e-5
)

# 学習開始
trainer.fit(
    model, 
    data_module
)

print('ベストモデルのファイル:', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失:', checkpoint.best_model_score)

# テスト
test = trainer.test(
    dataloaders=data_module.test_dataloader(), 
    ckpt_path=checkpoint.best_model_path
)

# ファインチューニングしたモデルのロード
finetuning_model = ModelForSequenceClassification_pl.load_from_checkpoint(
    checkpoint.best_model_path
)

# モデルの保存
finetuning_model.model_sc.save_pretrained(output_dir + '/model_finetuning/')
