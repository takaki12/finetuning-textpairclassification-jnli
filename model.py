# モデルモジュール

import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification

# モデル定義
class ModelForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=1e-5):
        """モデル設定

        Args:
            model_name (str): 事前学習済みモデル
            num_labels (int): 分類ラベル数
            lr (float): 学習率
        """

        super().__init__()
        self.save_hyperparameters()
        self.model_sc = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels
        )

    def training_step(self, batch, batch_idx):
        # スコアの計算
        output = self.model_sc(**batch)
        # ロスの計算
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # スコアの計算
        output = self.model_sc(**batch)
        # ロスの計算
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        # ラベルを取得
        labels = batch.pop('labels')
        # スコアの計算
        output = self.model_sc(**batch)
        # スコアをもとにラベル予測
        labels_predicted = output.logits.argmax(-1)

        # Accuracy
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0)
        self.log('accuracy', accuracy)

    # オプティマイザ
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    