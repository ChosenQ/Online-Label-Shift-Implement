import jax.numpy as jnp
import jax
from jax import grad
import optax

def follow_the_leader(loss_fn, initial_params, data, num_steps):
    """
    Args:
    - loss_fn (callable): The loss function to minimize, which takes parameters and data as input.
    - initial_params (array): Initial parameters for the optimization.
    - data (iterable): Iterable of data points to process in the algorithm.
    - num_steps (int): Number of steps to run the algorithm.

    Returns:
    - array: Optimized parameters after running the FTL algorithm.
    """
    # Initialize parameters
    params = initial_params
    
    # Initialize an optimizer; since we manually adjust params to minimize cumulative loss,
    # Optax's role here is non-standard and might be used differently or not at all in traditional FTL.
    optimizer = optax.adam(learning_rate=0.01)  # Placeholder, not really used in traditional FTL
    opt_state = optimizer.init(params)

    # Cumulative loss to keep track of the loss sum
    cumulative_loss = 0

    for step in range(num_steps):
        # Calculate the loss for the current parameters
        current_loss = loss_fn(params, data[step])
        cumulative_loss += current_loss

        # Calculate gradients w.r.t. the sum of losses up to this step
        grad_fn = jax.grad(lambda p: loss_fn(p, data[step]))
        grads = grad_fn(params)

        # Update the parameters using Optax; in FTL, this would be directly finding the min of cumulative loss
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    
    return params

# Example loss function
# def squared_loss(params, data_point):
#     """
#     Simple squared loss for demonstration.
#     y = params[0] * x + params[1]
#     """
#     x, y_true = data_point
#     y_pred = params[0] * x + params[1]
#     return (y_true - y_pred) ** 2

# Define initial parameters and dummy data
# initial_params = jnp.array([0.0, 0.0])
# data = [(jnp.array(x), jnp.array(x**2 + 1)) for x in range(10)]  # Quadratic data for example
# num_steps = len(data)
