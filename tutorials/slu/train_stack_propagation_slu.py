# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: demo.py
@time: 2021/03/02 21:19:27

这一行开始写关于本文件的说明与解释


'''

import tensorflow as tf
from cybo.data.dataloader import Dataloader
from cybo.data.vocabulary import Vocabulary
from cybo.models.stack_propagation_slu import StackPropagationSlu
from cybo.data.dataset_readers.slu_dataset_reader import SluDatasetReader, Tokenizer
from cybo.training.trainer import Trainer


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


dataset_reader = SluDatasetReader()
training_examples = dataset_reader.get_examples(
    filepath="demos/slu/dataset/atis/train.txt")
validation_examples = dataset_reader.get_examples(
    filepath="demos/slu/dataset/atis/dev.txt")
test_examples = dataset_reader.get_examples(
    filepath="demos/slu/dataset/atis/test.txt")

vocab = Vocabulary.from_examples(
    examples=training_examples + validation_examples + test_examples,
    non_padded_namespaces=["intent", "tags"])

training_features = dataset_reader.convert_examples_to_features(
    examples=training_examples, vocab=vocab, max_seq_length=32, verbose=True)
training_dataloader = Dataloader.from_features(
    training_features, batch_size=16)

validation_features = dataset_reader.convert_examples_to_features(
    examples=validation_examples, vocab=vocab, max_seq_length=32)
validation_dataloader = Dataloader.from_features(
    validation_features, batch_size=128)

model = StackPropagationSlu(
    vocab=vocab, embedding_dim=256, hidden_dim=256, dropout_rate=0.4)

trainer = Trainer(
    model=model, training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    checkpoint_path="./output_stack_propagation", epochs=200, monitor="nlu_acc",
    optimizer=tf.keras.optimizers.Adam())
trainer.train()
