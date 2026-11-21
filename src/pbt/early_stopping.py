"""Conventional validation-loss stopping, independent of TensorFlow."""


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.best_epoch = None
        self.wait = 0

    def observe(self, epoch, nll):
        improved = self.best is None or nll < self.best - self.min_delta
        if improved:
            self.best = nll
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
        return {
            "improved": improved,
            "best_epoch": self.best_epoch,
            "best_nll": self.best,
            "wait": self.wait,
            "stop": self.wait >= self.patience,
        }
