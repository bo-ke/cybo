# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: tensorboard.py
@time: 2021/05/12 01:18:35

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
import datetime

from cybo.training.utils import Mode


class TensorBoard:
    def __init__(self, logs_dir: str = 'logs/') -> None:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.train_log_dir = f"{logs_dir}{current_time}/train"
        self.dev_log_dir = f"{logs_dir}{current_time}/dev"
        self.test_log_dir = f"{logs_dir}{current_time}/test"

        self.train_summary_writer, self.dev_summary_writer, self.test_summary_writer = None, None, None

    def write_logs(self, mode: Mode, log_values, step):
        # tf.summary.scalar step需要是tf.int64
        step = tf.cast(step, dtype=tf.int64)
        if mode == Mode.train.value:
            if not self.train_summary_writer:
                self.train_summary_writer = tf.summary.create_file_writer(
                    self.train_log_dir)
            _writer = self.train_summary_writer
        elif mode == Mode.evaluate.value:
            if not self.dev_summary_writer:
                self.dev_summary_writer = tf.summary.create_file_writer(
                    self.dev_log_dir)
            _writer = self.dev_summary_writer
        elif mode == Mode.test.value:
            if not self.test_summary_writer:
                self.test_summary_writer = tf.summary.create_file_writer(
                    self.test_log_dir)
            _writer = self.test_summary_writer
        if _writer:
            with _writer.as_default():
                for k, v in log_values:
                    tf.summary.scalar(k, v, step=step)
