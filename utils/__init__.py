from typing import Callable

import numpy as np
from tic_tac_toe import get_available


def step_until_the_end_of_the_episode_and_generate_trajectory(
        s0: int,
        pi: np.ndarray,
        step_func: Callable,
        is_terminal_func: Callable,
        max_steps: int = 10
) -> ([int], [int], [int], [float]):
    s_list = []
    a_list = []
    s_p_list = []
    r_list = []
    st = s0
    actions = np.arange(pi.shape[1])
    step = 0
    while not is_terminal_func(st) and step < max_steps:
        at = np.random.choice(actions, p=pi[st])
        st_p, rt_p, terminal = step_func(st, at)
        s_list.append(st)
        a_list.append(at)
        s_p_list.append(st_p)
        r_list.append(rt_p)
        st = st_p
        step += 1

    return s_list, a_list, s_p_list, r_list

def step_until_the_end_of_the_episode_and_generate_trajectory_for_tic_tac_toe(
        s0: int,
        c0: int,
        pi: np.ndarray,
        step_func: Callable,
        is_terminal_func: Callable,
        max_steps: int = 10
) -> ([int], [int], [int], [float]):
    s_list = []
    a_list = []
    r_list = []
    c_list = []
    st = s0
    ct = c0
    actions = np.arange(pi.shape[2])
    step = 0
    while not is_terminal_func(st) and step < max_steps:
        at = np.random.choice(actions, p=pi[st, (ct.index(1) - 1)])
        st_p, rt_p, ct_p = step_func(st, at, ct)
        s_list.append(st)
        a_list.append(at)
        c_list.append(ct)
        r_list.append(rt_p)
        st = st_p
        step += 1

    return s_list, a_list, c_list, r_list