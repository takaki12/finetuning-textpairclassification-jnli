import pandas as pd
from sklearn.model_selection import KFold

from transformers import AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data_loader import DataModuleGenerator
from model import ModelForSequenceClassification_pl

# 事前学習済みモデルの指定 ---------------------
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# データの作成
train_datadict = {'text':[], 'label':[]}
val_datadict = {'text':[], 'label':[]}
test_datadict = {'text':[], 'label':[]}
# データフレーム化
train_df = pd.DataFrame(train_datadict)
val_df = pd.DataFrame(val_datadict)
test_df = pd.DataFrame(test_datadict)

# Checkpoint
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath= '/model/'
)

# Early_stopping
early_stopping = EarlyStopping(
    monitor='val_loss', 
    verbose=True, 
    mode="min", 
    patience=3
)

# Logger
tb_logger = pl_loggers.TensorBoardLogger(
    save_dir="/logs/"
)

trainer = pl.Trainer(
    accelerator='gpu', 
    devices=1,
    max_epochs=6,
    callbacks = [checkpoint, early_stopping],
    logger=tb_logger
)

# DataFrameを作成し、setupする
data_module = DataModuleGenerator(
    train_df=train_df,
    val_df=val_df, 
    test_df=test_df, 
    tokenizer=tokenizer, 
    max_length=512,
    batch_size={'train':32, 'val':256, 'test':256}
)
data_module.setup()

# モデル定義
model = ModelForSequenceClassification_pl(
    model_name, 
    num_labels=2, 
    lr=1e-5
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
print(f'Accuracy: {test[0]["accuracy"]:.2f}')

# ファインチューニングしたモデルのロード
finetuning_model = ModelForSequenceClassification_pl.load_from_checkpoint(
    checkpoint.best_model_path
)

# モデルの保存
finetuning_model.model_sc.save_pretrained('/model_finetuning')
