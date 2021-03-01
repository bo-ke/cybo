import tensorflow as tf
from cybo_dataloader import Dataloader
from cybo_vocabulary import Vocabulary
from cybo_model import StackPropagationSlu
from intent_slot_dataset_reader import IntentSlotDatasetReader, Tokenizer
from cybo_trainer import Trainer

dataset_reader = IntentSlotDatasetReader(tokenizer=Tokenizer())
examples = dataset_reader.get_examples(
    data_filepath="./data/map/map_first_part.csv")
print(examples[0])
vocab = Vocabulary.from_examples(
    examples=examples, non_padded_namespaces=["intent"])
# features = dataset_reader.convert_examples_to_features(
#     examples=examples, vocab=vocab, return_generator=True)
features = dataset_reader.convert_examples_to_features(
    examples=examples, vocab=vocab, max_seq_length=32)
dataloader = Dataloader.from_features(features, batch_size=128)
# dataloader = Dataloader.from_features_generator(
#     features, generator_size=len(examples),
#     output_types=dataset_reader.return_types, batch_size=4)

model = StackPropagationSlu(
    vocab_size=vocab.get_vocab_size(namespace="text"),
    embedding_dim=256, hidden_dim=256, dropout_rate=0.4,
    intent_size=vocab.get_vocab_size(namespace="intent"),
    slot_size=vocab.get_vocab_size(namespace="tags"))

trainer = Trainer(model=model, training_dataloader=dataloader,
                  checkpoint_path="./output_map", epochs=200,
                  optimizer=tf.keras.optimizers.Adam())
trainer.train()
