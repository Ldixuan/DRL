import numpy as np
from copy import copy

# C = np.array([1, 0, 0], [0 ,1, 0], [0, 0, 1]) # 1 : X, 2:O
# A = np.arange(9)
# T = np.array([])

S = []
A = np.arange(9)

MY_COLOR = np.array([0, 1, 0])
ADVERSARY_COLOR = np.array([0, 0, 1])
EMPTY_CELL = np.array([1, 0, 0])

def init_state(state = S, cell_list = [], cell_index = 0):
    if(cell_index == 9):
        state.append(cell_list)
        return
    for i in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        init_state(state, copy(cell_list + [i]),cell_index + 1)
    return

init_state()
S = np.asarray(S)


s_start = np.where(S == np.array([[1, 0, 0] for i in range(9)]))[0][0]
c_start = [0, 1, 0]

def get_available(state):
    ret = []
    state = S[state]
    for i in range(len(state)):
        if(np.array_equal(state[i], EMPTY_CELL)):
            ret.append(i)
    return np.array(ret)

def step(state, action : int, color):
    assert action >= 0 and action < 9, "invalid"
    assert not is_terminal(state), "game is finished"
    assert np.array_equal(S[state, action], EMPTY_CELL), "cell is not empty"
    reward = 1
    
    state = np.copy(S[state])
    state[action] = color
    next_state = np.copy(state)
    next_color = [[0, 1, 1][i] - color[i] for i in range(3)]
    next_state_index = np.where(S == next_state)[0][0]
    
    if(len(get_available(next_state)) == 0):
        reward = 1
        return next_state_index, reward, next_color
    elif(is_terminal_by(next_state, color)):
        reward = 2
        return next_state_index, reward, next_color
    elif(is_terminal_by(next_state, next_color)):
        reward = -1
        return next_state_index, reward, next_color
    
    
    
    return next_state_index, reward, next_color
    
def reset():
    return s_start

def is_terminal_by(state_index : int, color = MY_COLOR):
    state = S[state_index]
    return np.array([len(np.unique(state[[0,1,2]], axis=0)) == 1 and np.array_equal(state[0],color),
    len(np.unique(state[[3, 4, 5]], axis=0)) == 1 and np.array_equal(state[3], color),
    len(np.unique(state[[6, 7, 8]], axis=0)) == 1 and np.array_equal(state[6], color),
    len(np.unique(state[[0, 3, 6]], axis=0)) == 1 and np.array_equal(state[0], color),
    len(np.unique(state[[1, 4, 7]], axis=0)) == 1 and np.array_equal(state[1], color),
    len(np.unique(state[[2, 5, 8]], axis=0)) == 1 and np.array_equal(state[2], color),
    len(np.unique(state[[0, 4, 8]], axis=0)) == 1 and np.array_equal(state[0], color),
    len(np.unique(state[[2, 4, 6]], axis=0)) == 1 and np.array_equal(state[2], color),]).any() or (len(get_available(state_index)) == 0)

def is_terminal(state : int):
    return is_terminal_by(state, MY_COLOR) or is_terminal_by(state, ADVERSARY_COLOR)

