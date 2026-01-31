import tensorflow as tf

# Define sample tensors (e.g., three 1D tensors of length 2)
A = tf.constant([1, 2])
B = tf.constant([3, 4])
C = tf.constant([5, 6])

# Stack them along axis 0 (default)
# Resulting shape: (3, 2)
stacked_0 = tf.stack([A, B, C], axis=0)

# Stack them along axis 1
# Resulting shape: (2, 3)
stacked_1 = tf.stack([A, B, C], axis=1)

print("Axis 0:\n", stacked_0.numpy())
print("Axis 1:\n", stacked_1.numpy())
