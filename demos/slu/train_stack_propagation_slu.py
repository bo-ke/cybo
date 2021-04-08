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
# from cybo.models.stack_propagation_slu import StackPropagationSlu
from cybo.data.dataset_readers.slu_dataset_reader import SluDatasetReader, Tokenizer
from cybo.training.trainer import Trainer

dataset_reader = SluDatasetReader(tokenizer=Tokenizer())
training_examples = dataset_reader.get_examples(
    filepath="demos/slu/dataset/atis/train.txt")
validation_examples = dataset_reader.get_examples(
    filepath="demos/slu/dataset/atis/dev.txt")
test_examples = dataset_reader.get_examples(
    filepath="demos/slu/dataset/atis/test.txt")


vocab = Vocabulary.from_examples(
    examples=training_examples + validation_examples,
    non_padded_namespaces=["intent", "tags"])
# features = dataset_reader.convert_examples_to_features(
#     examples=examples, vocab=vocab, return_generator=True)
training_features = dataset_reader.convert_examples_to_features(
    examples=training_examples, vocab=vocab, max_seq_length=32, verbose=True)
training_dataloader = Dataloader.from_features(
    training_features, batch_size=128)

validation_features = dataset_reader.convert_examples_to_features(
    examples=validation_examples, vocab=vocab, max_seq_length=32)
validation_dataloader = Dataloader.from_features(
    validation_features, batch_size=128)
# dataloader = Dataloader.from_features_generator(
#     features, generator_size=len(examples),
#     output_types=dataset_reader.return_types, batch_size=4)

# model = StackPropagationSlu(
#     vocab_size=vocab.get_vocab_size(namespace="text"),
#     embedding_dim=256, hidden_dim=256, dropout_rate=0.4,
#     intent_size=vocab.get_vocab_size(namespace="intent"),
#     slot_size=vocab.get_vocab_size(namespace="tags"))

# trainer = Trainer(model=model, training_dataloader=training_dataloader,
#                   checkpoint_path="./output_map", epochs=200,
#                   optimizer=tf.keras.optimizers.Adam())
# trainer.train()
