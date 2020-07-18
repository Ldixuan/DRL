import numpy as np
from typing import Callable
from copy import copy

from policies import tabular_random_uniform_policy
from policies import tabular_random_uniform_policy_for_tic_tac_toe

from utils import step_until_the_end_of_the_episode_and_generate_trajectory
from utils import step_until_the_end_of_the_episode_and_generate_trajectory_for_tic_tac_toe


def iterative_policy_evaluation(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        Pi: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.00001
) -> np.ndarray:
    assert theta > 0
    assert 0 <= gamma <= 1
    V = np.random.random((S.shape[0],))
    V[T] = 0.0
    while True:
        delta = 0
        for s in S:
            v_temp = V[s]
            new_v = 0
            for a in A:
                for s_p in S:
                    new_v += Pi[s, a] * P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
            V[s] = new_v
            delta = np.maximum(delta, np.abs(v_temp - new_v))
        if delta < theta:
            break
    return V


def policy_iteration(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.00001
) -> (np.ndarray, np.ndarray):
    Pi = tabular_random_uniform_policy(S.shape[0], A.shape[0])
    while True:
        V = iterative_policy_evaluation(S, A, P, T, Pi, gamma, theta)
        policy_stable = True
        for s in S:
            old_action = np.argmax(Pi[s])
            best_action = 0
            best_action_score = -9999999999
            for a in A:
                tmp_sum = 0
                for s_p in S:
                    tmp_sum += Pi[s, a] * P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
                if tmp_sum > best_action_score:
                    best_action = a
                    best_action_score = tmp_sum
            Pi[s] = 0.0
            Pi[s, best_action] = 1.0
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    V = iterative_policy_evaluation(S, A, P, T, Pi, gamma, theta)
    return V, Pi

def value_iteration(
    S: np.ndarray,
    A: np.ndarray,
    P: np.ndarray,
    T: np.ndarray,
    gamma: float = 0.99,
    theta: float = 0.00001
) -> np.ndarray:
    V = np.random.random((S.shape[0],))
    V[T] = 0.0
    while True:
        delta = 0
        for s in S:
            v_temp = V[s]
            v_max = -99999999
            for a in A:
                v_max_temp = 0
                for s_p in S:
                    v_max_temp += P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
                if(v_max < v_max_temp):
                    v_max = v_max_temp
            V[s] = v_max
            delta = np.maximum(delta, np.abs(v_temp - v_max))

        if delta < theta:
            break
    
    Pi = tabular_random_uniform_policy(S.shape[0], A.shape[0])
    for s in S:
        old_action = np.argmax(Pi[s])
        best_action = 0
        best_action_score = -9999999999
        for a in A:
            tmp_sum = 0
            for s_p in S:
                tmp_sum += Pi[s, a] * P[s, a, s_p, 0] * (
                        P[s, a, s_p, 1] + gamma * V[s_p]
                )
            if tmp_sum > best_action_score:
                best_action = a
                best_action_score = tmp_sum
        Pi[s] = 0.0
        Pi[s, best_action] = 1.0
    return V, Pi
    



def monte_carlo_es(
        states_count: int,
        actions_count: int,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99
) -> (np.ndarray, np.ndarray):
    pi = tabular_random_uniform_policy(states_count, actions_count)
    states = np.arange(states_count)
    actions = np.arange(actions_count)

    Q = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.0
            pi[s, :] = 0.0

    returns = np.zeros((states_count, actions_count))
    returns_count = np.zeros((states_count, actions_count))

    for episode_id in range(max_episodes):
        s0 = np.random.choice(states)

        if is_terminal_func(s0):
            episode_id -= 1
            continue

        a0 = np.random.choice(actions)

        s1, r1, terminal = step_func(s0, a0)

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_generate_trajectory(s1, pi, step_func,
                                                                                              is_terminal_func,
                                                                                              max_steps_per_episode)
        s_list.insert(0, s0)
        a_list.insert(0, a0)
        r_list.insert(0, r1)

        G = 0.0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue

            returns[st, at] += G
            returns_count[st, at] += 1
            Q[st, at] = returns[st, at] / returns_count[st, at]
            pi[st, :] = 0.0
            pi[st, np.argmax(Q[st, :])] = 1.0
    return Q, pi


def monte_carlo_es_for_tic_tac_toe(
        s0 : int,
        c0 : np.ndarray,
        states_count: int,
        actions_count: int,
        step_func: Callable,
        is_terminal_func: Callable,
        get_actions_func:Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99
) -> (np.ndarray, np.ndarray):
    pi = tabular_random_uniform_policy_for_tic_tac_toe(states_count, actions_count)
    
    states = np.arange(states_count)
    actions = get_actions_func(s0)

    Q = np.random.random((states_count, 2, actions_count))

    for s in range(states_count):
        if is_terminal_func(s):
            Q[s, :] = 0.0
            pi[s, :] = 0.0
            
        availble_cells = get_actions_func(s)
        for c in range(2):
            for a in range(actions_count):
                if(a not in availble_cells):
                    pi[s, c, a] = 0.0
                    Q[s, c, a] = 0.0

    returns = np.zeros((states_count, 2, actions_count))
    returns_count = np.zeros((states_count, 2, actions_count))

    for episode_id in range(max_episodes):
        s_temp = s0
        
        a0 = np.random.choice(actions)
        
        s1, r1, c1 = step_func(s_temp, a0, c0)

        s_list, a_list, c_list , r_list = step_until_the_end_of_the_episode_and_generate_trajectory_for_tic_tac_toe(s1, c1, pi, step_func,
                                                                                                  is_terminal_func,
                                                                                                  max_steps_per_episode)
        s_list.insert(0, s_temp)
        a_list.insert(0, a0)
        c_list.insert(0, c0)
        r_list.insert(0, 1)

        G = 0.0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            ct = c_list[t]
            # cr = [[0, 1, 1][i] - ct[i] for i in range(3)]
            if (st, ct, at) in zip(s_list[0:t], c_list[0:t], a_list[0:t]):
                continue

            returns[st, (ct.index(1) - 1), at] += G
            returns_count[st, (ct.index(1) - 1), at] += 1
            Q[st,(ct.index(1) - 1), at] = returns[st, (ct.index(1) - 1), at] / returns_count[st, (ct.index(1) - 1), at]
            pi[st, (ct.index(1) - 1), :] = 0.0
            pi[st, (ct.index(1) - 1), np.argmax(Q[st, (ct.index(1) - 1), :])] = 1.0
    return Q, pi


def on_policy_first_visit_monte_carlo(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        epsilon: float = 0.1
) -> (np.ndarray, np.ndarray):
    pi = tabular_random_uniform_policy(states_count, actions_count)
    states = np.arange(states_count)
    actions = np.arange(actions_count)

    Q = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.0
            pi[s, :] = 0.0

    returns = np.zeros((states_count, actions_count))
    returns_count = np.zeros((states_count, actions_count))

    for episode_id in range(max_episodes):
        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_generate_trajectory(s0, pi, step_func,
                                                                                              is_terminal_func,
                                                                                              max_steps_per_episode)

        G = 0.0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue

            returns[st, at] += G
            returns_count[st, at] += 1
            Q[st, at] = returns[st, at] / returns_count[st, at]
            pi[st, :] = epsilon / actions_count
            pi[st, np.argmax(Q[st, :])] = 1.0 - epsilon + epsilon / actions_count
    return Q, pi


def off_policy_monte_carlo_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        step_func: Callable,
        is_terminal_func: Callable,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_greedy_behaviour_policy: bool = False
) -> (np.ndarray, np.ndarray):
    b = tabular_random_uniform_policy(states_count, actions_count)
    pi = tabular_random_uniform_policy(states_count, actions_count)
    states = np.arange(states_count)

    Q = np.random.random((states_count, actions_count))

    for s in states:
        if is_terminal_func(s):
            Q[s, :] = 0.0
            pi[s, :] = 0.0
        pi[s, :] = 0
        pi[s, np.argmax(Q[s, :])] = 1.0

    C = np.zeros((states_count, actions_count))

    for episode_id in range(max_episodes):
        if epsilon_greedy_behaviour_policy:
            for s in states:
                b[s, :] = epsilon / actions_count
                b[s, np.argmax(Q[s, :])] = 1.0 - epsilon + epsilon / actions_count

        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_generate_trajectory(s0, b, step_func,
                                                                                              is_terminal_func,
                                                                                              max_steps_per_episode)

        G = 0.0
        W = 1
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            C[st, at] += W

            Q[st, at] += W / C[st, at] * (G - Q[st, at])
            pi[st, :] = 0
            pi[st, np.argmax(Q[st, :])] = 1.0
            if np.argmax(Q[st, :]) != at:
                break
            W = W / b[st, at]
    return Q, pi