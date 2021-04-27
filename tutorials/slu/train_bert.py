# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: train_bert.py
@time: 2021/04/25 00:25:12

这一行开始写关于本文件的说明与解释


'''
from cybo.training.utils import test
from cybo.training.trainer import Trainer
from cybo.data.dataset_readers.transformers_slu_dataset_reader import TransformersSluDatasetReader
from cybo.data.tokenizers.transformers_bert_tokenizer import TransformersBertTokenizer
from cybo.models.bert_slu import BertSlu
from cybo.modules.transformers_pretrained_layer import TransformersPretrainedLayer
from cybo.data.vocabulary import Vocabulary
from cybo.data.dataloader import Dataloader
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


dataset_reader = TransformersSluDatasetReader(
    tokenizer=TransformersBertTokenizer.from_pretrained("bert-base-uncased"))
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

model = BertSlu(
    vocab=vocab,
    pretrained_layer=TransformersPretrainedLayer(
        pretrained_model="bert-base-uncased"),
    dropout_rate=0.4)


def train():
    trainer = Trainer(
        model=model, training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        checkpoint_path="./output_atis_bert", epochs=100,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        monitor="nlu_acc")
    trainer.train()


def test_model():
    test_features = dataset_reader.convert_examples_to_features(
        examples=test_examples, vocab=vocab, max_seq_length=32)
    test_dataloader = Dataloader.from_features(test_features, batch_size=128)
    print(test(model=model, dataloader=test_dataloader,
               checkpoint_dir="./output_atis_bert"))


if __name__ == "__main__":
    train()
    test_model()
