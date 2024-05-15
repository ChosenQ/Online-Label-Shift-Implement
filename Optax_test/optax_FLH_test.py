import jax.numpy as jnp
import optax

def follow_the_leading_history(T, initial_experts, loss_function, learning_rate):
    """
    Args:
        T (int): Number of time steps to iterate over.
        initial_experts (list): List of initial expert prediction functions.
        loss_function (callable): Function to calculate the loss given predictions and actual outcomes.
        learning_rate (float): Learning rate for updates.

    Returns:
        List of updated probability distributions over experts at each time step.
    """
    num_experts = len(initial_experts)
    # Initialize probabilities: Start with equal probabilities for each initial expert
    probabilities = jnp.ones(num_experts) / num_experts

    # Record of probabilities over time
    probability_history = [probabilities]

    # Simulate data and outcomes - this should be replaced by real predictions and feedback
    outcomes = jnp.random.rand(T)  # Dummy outcomes for simulation purposes

    # Iterate over each time step
    for t in range(T):
        # Predictions by each expert
        predictions = jnp.array([expert(outcomes[max(0, t-1)]) for expert in initial_experts])

        # Calculate loss for each expert
        losses = loss_function(predictions, outcomes[t])

        # Exponential update of probabilities
        updates = jnp.exp(-learning_rate * losses)
        probabilities *= updates
        probabilities /= jnp.sum(probabilities)  # Normalize to maintain as a probability distribution

        # Optionally add a new expert or adjust probabilities - omitted for simplicity

        # Pruning step could be implemented here - omitted for simplicity

        # Store updated probabilities
        probability_history.append(probabilities)

    return probability_history


#Example of Exp
# Define dummy experts and a loss function for demonstration
# def dummy_expert_1(last_outcome):
#     return last_outcome + 0.1  # Simple strategy adding 0.1

# def dummy_expert_2(last_outcome):
#     return last_outcome - 0.1  # Simple strategy subtracting 0.1

# def squared_loss(predictions, actual):
#     return (predictions - actual) ** 2

# probabilities_over_time = follow_the_leading_history(
#     T=10,
#     initial_experts=[dummy_expert_1, dummy_expert_2],
#     loss_function=squared_loss,
#     learning_rate=0.01
# )
# print(probabilities_over_time)