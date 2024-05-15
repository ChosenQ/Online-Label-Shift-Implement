import optax

class Config:
    def __init__(self, optimizer, lr, weight_decay, **optimizer_kwargs):
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_kwargs = optimizer_kwargs

def initialize_optimizer(config):
    # Define mapping from configuration to Optax optimizers
    optimizer_mapping = {
        'SGD': optax.sgd(learning_rate=config.lr, **config.optimizer_kwargs),
        'Adam': optax.adam(learning_rate=config.lr, weight_decay=config.weight_decay, **config.optimizer_kwargs),
        'AdamW': optax.adamw(learning_rate=config.lr, weight_decay=config.weight_decay, **config.optimizer_kwargs)
    }

    # Get optimizer based on config
    if config.optimizer in optimizer_mapping:
        return optimizer_mapping[config.optimizer]
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized.')