import tensorflow as tf

from cybo_dataloader import Dataloader
from cybo.models.model import Model


def evaluate(model, dataloader):
    loss_metric = tf.keras.metrics.Mean(name="loss")
    model.get_metrics(reset=True)
    for batch in dataloader:
        output_dict = model(**batch, training=False)
        loss_metric.update_state(output_dict["loss"])
    metrics_to_return = model.get_metrics()
    metrics_to_return["loss"] = loss_metric.result().numpy()
    return metrics_to_return


class Trainer():
    def init(self,
             model: Model,
             training_dataloader: Dataloader,
             optimizer: tf.keras.optimizers.Optimizer,
             epochs: int,
             checkpoint_path: str,
             validation_dataloader: Dataloader = None,
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
            for batch in self.training_dataloader:
                self.train_step(batch)
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

    # @tf.function()
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            output_dict = self.model(**batch, training=True)
        gradients = tape.gradient(
            output_dict["loss"],
            self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.loss_metric.update_state(output_dict["loss"])
        # return output_dict
