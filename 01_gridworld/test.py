import numpy as np

q_values = np.zeros((1, 4))
print(q_values)
max_q = np.max(q_values)
best_actions = np.flatnonzero(q_values == max_q)

print(best_actions)
