import numpy as np

def fair_sharer(values, num_iterations, share=0.1):
    """
    Distributes a fraction of the highest value to its left and right neighbors
    in a circular array over a specified number of iterations.

    Args:
    values (np.array): A 1D numpy array of integer values.
    num_iterations (int): The number of iterations to run.
    share (float): The fraction of value to be shared with each neighbor.

    Returns:
    np.array: The resulting array after distribution.
    """
    for _ in range(num_iterations):
        # Find the index of the highest value
        max_index = np.argmax(values)

        # Calculate the amount to be shared
        shared_amount = values[max_index] * share

        # Identify left and right neighbors
        left_neighbor = (max_index - 1) % len(values)
        right_neighbor = (max_index + 1) % len(values)

        # Distribute the shares
        values[left_neighbor] += shared_amount / 2
        values[right_neighbor] += shared_amount / 2

        # Subtract the shared amount from the original value
        values[max_index] -= shared_amount

    return values

# Example usage
values = np.random.randint(0, 1000, size=4)
num_iterations = 4

print("Original values:", values)
result = fair_sharer(values, num_iterations)
print("Values after distribution:", result)
