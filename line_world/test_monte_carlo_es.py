import sys

sys.path.append("C:/Users/49135/Documents/m1/deeplearning/projet-RDL")
from line_world import S, A, reset, step, is_terminal
from algorithms import monte_carlo_es
import time

t1 = time.time()
Q, Pi = monte_carlo_es(len(S), len(A),
                        step,
                        is_terminal,
                        max_episodes=10000,
                        max_steps_per_episode=100)
print(Q)
print(Pi)
print(f"le time d'execution : {time.time() - t1}")