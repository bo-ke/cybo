# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: torch_trainer.py
@time: 2021/06/22 21:21:14

这一行开始写关于本文件的说明与解释


'''
import numpy as np
import torch
from torch.utils.data import DataLoader
import pkbar


class Trainer():
    _device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self, model, epochs, training_dataloader: DataLoader,
            optimizer=torch.optim.Adam, lr=0.001,
            val_dataloader=None) -> None:

        self.epochs = epochs
        self.model = model.to(self._device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.training_dataloader = training_dataloader
        self.val_dataloader = val_dataloader if val_dataloader else training_dataloader

    def evaluate(self):
        self.model.eval()
        val_loss = []
        for batch in self.val_dataloader:
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.no_grad():
                output_dict = self.model(**batch)
                val_loss.append(output_dict["loss"].item())
        val_loss = np.mean(val_loss)
        val_metrics = self.model.get_metrics(reset=True)

        return val_loss, val_metrics

    def train_step(self, batch):
        self.optimizer.zero_grad()
        batch = {k: v.to(self._device) for k, v in batch.items()}
        output_dict = self.model(**batch)
        loss = output_dict["loss"]
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        for epoch in range(self.epochs):
            _metrics = self.model.get_metrics(reset=True)
            kbar = pkbar.Kbar(
                target=len(self.training_dataloader),
                epoch=epoch, num_epochs=self.epochs,
                stateful_metrics=list(_metrics))
            self.model.train()
            for i, batch in enumerate(self.training_dataloader):
                _loss = self.train_step(batch)
                _metrics = self.model.get_metrics(reset=True)
                kbar.update(i, values=[("loss", _loss)]+[(k, v)
                                                         for k, v in _metrics.items()])
            val_loss, val_metrics = self.evaluate()
            kbar.add(1, values=[("val_loss", val_loss)] +
                     [(f"val_{k}", v) for k, v in val_metrics.items()])
