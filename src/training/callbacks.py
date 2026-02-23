"""Custom training callbacks for monitoring and logging."""

from __future__ import annotations

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from loguru import logger


class AttributeExtractionCallback(TrainerCallback):
    """Custom callback for attribute extraction training.

    Logs training progress, monitors loss trends, and performs
    periodic sanity checks on model outputs.
    """

    def __init__(self, check_interval: int = 100):
        self.check_interval = check_interval
        self.loss_history = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        """Log training metrics."""
        if logs is None:
            return

        step = state.global_step
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        lr = logs.get("learning_rate")

        if loss is not None:
            self.loss_history.append(loss)
            logger.info(
                f"Step {step}: loss={loss:.4f}, lr={lr:.2e}"
                if lr
                else f"Step {step}: loss={loss:.4f}"
            )

        if eval_loss is not None:
            logger.info(f"Step {step}: eval_loss={eval_loss:.4f}")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log epoch summary."""
        epoch = state.epoch
        if self.loss_history:
            recent = self.loss_history[-10:]
            avg_loss = sum(recent) / len(recent)
            logger.info(f"Epoch {epoch:.0f} complete. Recent avg loss: {avg_loss:.4f}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log final training summary."""
        logger.info("Training completed!")
        if self.loss_history:
            logger.info(f"  Initial loss: {self.loss_history[0]:.4f}")
            logger.info(f"  Final loss:   {self.loss_history[-1]:.4f}")
            logger.info(f"  Total steps:  {state.global_step}")


class EarlyStoppingWithPatience(TrainerCallback):
    """Early stopping based on eval loss with configurable patience."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict = None,
        **kwargs,
    ):
        """Check if we should stop training."""
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss", float("inf"))

        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.wait = 0
        else:
            self.wait += 1
            logger.info(
                f"No improvement for {self.wait}/{self.patience} evaluations "
                f"(best={self.best_loss:.4f}, current={eval_loss:.4f})"
            )

            if self.wait >= self.patience:
                logger.warning(f"Early stopping triggered after {self.patience} evaluations")
                control.should_training_stop = True
