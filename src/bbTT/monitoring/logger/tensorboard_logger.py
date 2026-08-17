import os
import pathlib


class TensorboardLogger():
    def __init__(self, name=None, path=None):
        self.name = name
        self.log_dir = pathlib.Path(os.environ["TENSORBOARD_DIR"])
        self.path = self.logger_path() if path is None else self.log_dir / path
        self.writer = self.create_tensorboard_writer(log_dir=self.path)

    def log_loss(self, values, step):
        self.writer.add_scalars("Loss", values, global_step=step)

    def log_scalar(self, values, step, name):
        self.writer.add_scalars(name, values, global_step=step)

    def log_lr(self, optimizer, step):
        value = optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Learning_Rate", value, step)

    def log_precision(self, values, step, mode="train"):
        acc = {k : v["precision"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} Precision", acc, step)

    def log_f1(self, values, step, mode="train"):
        acc = {k : v["f1"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} F1-Score", acc, step)

    def log_sensitivity(self, values, step, mode="train"):
        acc = {k: v["sensitivity"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} Recall", acc, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)

    def logger_path(self):
        from time import localtime, strftime
        t = strftime("%Y_%m_%d-%H_%M_%S", localtime())
        new_stem = f"{t}-{self.name}"
        logger_path = self.log_dir / new_stem
        return logger_path

    def create_tensorboard_writer(self, config=None, log_dir=None):
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
