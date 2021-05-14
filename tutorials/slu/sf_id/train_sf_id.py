# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: train_sf_id.py
@time: 2021/04/24 15:45:06

这一行开始写关于本文件的说明与解释


'''
from cybo.training.utils import test
from cybo.training.trainer import Trainer
from cybo.data.dataset_readers.slu_dataset_reader import SluDatasetReader
from cybo.models.sf_id import SfId
from cybo.data.vocabulary import Vocabulary
from cybo.data.dataloader import Dataloader
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


dataset_reader = SluDatasetReader()
training_examples = dataset_reader.get_examples(
    filepath="tutorials/slu/dataset/atis/train.txt")
validation_examples = dataset_reader.get_examples(
    filepath="tutorials/slu/dataset/atis/dev.txt")
test_examples = dataset_reader.get_examples(
    filepath="tutorials/slu/dataset/atis/test.txt")

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
test_features = dataset_reader.convert_examples_to_features(
    examples=test_examples, vocab=vocab, max_seq_length=32)
test_dataloader = Dataloader.from_features(test_features, batch_size=128)

model = SfId(vocab=vocab,
             embedding_dim=256, hidden_dim=256, dropout_rate=0.4,
             iteration_num=1,
             use_crf=True)


def train():
    trainer = Trainer(model=model, training_dataloader=training_dataloader,
                      validation_dataloader=validation_dataloader,
                      checkpoint_path="./output_atis_sf_id", epochs=100,
                      optimizer=tf.keras.optimizers.Adam(),
                      patience=15, use_tensorboard=True,
                      monitor="nlu_acc")
    trainer.train()


def test_model():
    print(test(model=model, dataloader=test_dataloader,
               checkpoint_dir="./output_atis_sf_id"))


if __name__ == "__main__":
    train()
    test_model()
