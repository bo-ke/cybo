# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: trainer.py
@time: 2021/02/06 23:51:18

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


from cybo.models.model import Model
from cybo.training.utils import evaluate
from cybo.data.dataloader import Dataloader


class Trainer:
    def __init__(self,
                 model: Model,
                 training_dataloader: Dataloader,
                 validation_dataloader: Dataloader,
                 optimizer: tf.keras.optimizers.Optimizer,
                 epochs: int,
                 checkpoint_path: str,
                 patience: int = None,
                 save_weights_only: bool = True,
                 monitor: str = "overall_acc"
                 ) -> None:

        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader or training_dataloader
        self.optimizer = optimizer
        self.epochs = epochs

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

        # self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.monitor = monitor

    def train(self):
        ckpt = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer, epoch=tf.Variable(1))
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, self.checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            tf.print("restore from latest checkpoint succeed !")

        best_acc = 0.0
        for epoch in tf.range(ckpt.epoch, self.epochs+1):
            tf.print(f"Epoch {epoch}/{self.epochs}:")
            # 更新ckpt中epoch值
            ckpt.epoch.assign_add(1)
            metrics = self.model.get_metrics(reset=True)
            self.loss_metric.reset_states()

            bar = tf.keras.utils.Progbar(
                len(self.training_dataloader),
                unit_name="sample",
                stateful_metrics=["loss"] + list(metrics.keys()))

            log_values = []
            for x, y in self.training_dataloader:
                y_pred = self.train_step(x, y)
                self.model.update_metrics_state(y, y_pred)
                log_values.append(("loss", self.loss_metric.result().numpy()))
                log_values.extend([(k, v)
                                   for k, v in self.model.get_metrics().items()])
                bar.add(self.training_dataloader.batch_size, log_values)
            evaluate_metrics = evaluate(
                model=self.model, dataloader=self.validation_dataloader)
            tf.print("validation result - " +
                     " - ".join([f"{k}: {v}" for k, v in evaluate_metrics.items()]))
            if evaluate_metrics.get(self.monitor, 1.0) >= best_acc:
                ckpt_save_path = ckpt_manager.save()
                tf.print(
                    f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")
                best_acc = evaluate_metrics.get(self.monitor, 1.0)
            else:
                tf.print(f"validation {self.monitor} is not improved")

    @tf.function()
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(**x, training=True)
            loss = self.model.get_loss(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.loss_metric.update_state(loss)
        return y_pred
