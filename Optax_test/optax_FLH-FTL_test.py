import jax.numpy as jnp
from jax import grad, jit
import optax
from functools import partial


grad_fn = jit(grad(loss))
model=model

# FLH-FTL 
class FLHFTL:
    def __init__(self, num_experts, dim, alpha):
        self.alpha = alpha
        self.weights = jnp.ones(num_experts) / num_experts 
        self.params = [jnp.zeros(dim) for _ in range(num_experts)]  
        self.optimizers = [optax.adam(1e-2) for _ in range(num_experts)]  
        self.opt_states = [opt.init(param) for opt, param in zip(self.optimizers, self.params)]

    def update(self, x, y):
        predictions = jnp.array([model(p, x) for p in self.params])
        losses = jnp.array([loss_fn(p, x, y) for p in self.params])
        gradients = jnp.array([grad_fn(p, x, y) for p in self.params])
        
        for i, (grad, opt, state) in enumerate(zip(gradients, self.optimizers, self.opt_states)):
            updates, new_state = opt.update(grad, state)
            self.params[i] -= updates
            self.opt_states[i] = new_state

        weight_updates = jnp.exp(-self.alpha * losses)
        self.weights *= weight_updates
        self.weights /= jnp.sum(self.weights)  

    def predict(self, x):
        predictions = jnp.array([model(p, x) for p in self.params])
        return jnp.dot(self.weights, predictions)
    
    
# num_experts=num_experts, dim=dim, alpha=alpha
# flh_ftl = FLHFTL(num_experts=num_experts, dim=dim, alpha=alpha)
# x_data, y_data=x_data, y_data
# flh_ftl.update(x_data, y_data)
# prediction = flh_ftl.predict(x_data)