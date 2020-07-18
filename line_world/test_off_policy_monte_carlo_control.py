import sys

sys.path.append("C:/Users/49135/Documents/m1/deeplearning/projet-RDL")
from line_world import S, A, reset, step, is_terminal
from algorithms import off_policy_monte_carlo_control
import time

t1 = time.time()
Q, Pi = off_policy_monte_carlo_control(len(S), len(A),
                                            reset,
                                            step,
                                            is_terminal,
                                            max_episodes=10000, max_steps_per_episode=100)
print(Q)
print(Pi)
print(f"le time d'execution : {time.time() - t1}")