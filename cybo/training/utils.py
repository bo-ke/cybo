# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: utils.py
@time: 2021/02/23 00:58:46

这一行开始写关于本文件的说明与解释


'''
from enum import Enum
import tensorflow as tf
from cybo.data.dataloader import Dataloader
from cybo.models.model import Model


class Mode(Enum):
    """运行模式
    """
    train = "train"
    evaluate = "dev"
    test = "test"


def evaluate(model: Model, dataloader: Dataloader):
    loss_metric = tf.keras.metrics.Mean(name="loss")
    model.get_metrics(reset=True)
    for batch in dataloader:
        output_dict = model(**batch, training=False)
        loss_metric.update_state(output_dict["loss"])
    metrics_to_return = model.get_metrics(training=False, reset=True)
    metrics_to_return["loss"] = loss_metric.result().numpy()
    return metrics_to_return


def test(model: Model, dataloader: Dataloader, checkpoint_dir: str = None):
    if checkpoint_dir is not None:
        # load model weights from checkpoint_dir
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(tf.train.latest_checkpoint(
            checkpoint_dir=checkpoint_dir)).expect_partial()
    metrics = evaluate(model=model, dataloader=dataloader)
    return metrics
