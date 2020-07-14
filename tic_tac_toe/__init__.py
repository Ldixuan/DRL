import numpy as np

# C = np.array([1, 0, 0], [0 ,1, 0], [0, 0, 1]) # 1 : X, 2:O
# A = np.arange(9)
# T = np.array([])

S = []
A = np.arange(9)

def init_state(state = S, cell_list = [], cell_index = 0):
    if(cell_index == 9):
        state.append(cell_list)
        return
    for i in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        init_state(state, cell_list + [i],cell_index + 1)
    return

init_state()

S = np.array(S)
print(S.shape)

def get_available(state):
    ret = []
    for i in range(len(state)):
        if(np.array_equal(state[i], [1, 0, 0])):
            ret.append(i)
    return ret

def step(state, action : int, color):
    assert action >= 0 and action < 9, "invalid"
    assert not is_terminal(state), "game is finished"
    assert np.array_equal(state[action], [1,0,0]), "cell is not empty"
    state[action] = color
    next_state = np.copy(state)
    color = [[0, 1, 1][i] - color[i] for i in range(3)]
    return next_state, color

def is_terminal(state):
    return np.array([len(np.unique(state[[0,1,2]], axis=0)) == 1 
    and (np.array_equal(state[0],np.array([0, 1, 0])) or np.array_equal(state[0],np.array([0, 0, 1]))),
    len(np.unique(state[[3, 4, 5]], axis=0)) == 1
    and (np.array_equal(state[3],np.array([0, 1, 0])) or np.array_equal(state[3],np.array([0, 0, 1]))),
    len(np.unique(state[[6, 7, 8]], axis=0)) == 1
    and (np.array_equal(state[6],np.array([0, 1, 0])) or np.array_equal(state[6],np.array([0, 0, 1]))),
    len(np.unique(state[[0, 3, 6]], axis=0)) == 1
    and (np.array_equal(state[0],np.array([0, 1, 0])) or np.array_equal(state[0],np.array([0, 0, 1]))),
    len(np.unique(state[[1, 4, 7]], axis=0)) == 1
    and (np.array_equal(state[1],np.array([0, 1, 0])) or np.array_equal(state[1],np.array([0, 0, 1]))),
    len(np.unique(state[[2, 5, 8]], axis=0)) == 1
    and (np.array_equal(state[2],np.array([0, 1, 0])) or np.array_equal(state[2],np.array([0, 0, 1]))),
    len(np.unique(state[[0, 4, 8]], axis=0)) == 1
    and (np.array_equal(state[0],np.array([0, 1, 0])) or np.array_equal(state[0],np.array([0, 0, 1]))),
    len(np.unique(state[[2, 4, 6]], axis=0)) == 1
    and (np.array_equal(state[2],np.array([0, 1, 0])) or np.array_equal(state[0],np.array([0, 0, 1]))),
    ]).any() or (len(get_available(state)) == 0)


# board = np.array([[1, 0, 0] for i in range(9)])
# S.append(board)

# for i in range(9):
#     board[i] = [0,1,0]
#     S.append(np.copy(board))
#     board[i] = [0,0,1]
#     S.append(np.copy(board))

# def reset_s(board = np.array([[1, 0, 0] for i in range(9)]), color = [0, 1, 0]):
#     S.append(board)
#     print(len(S))
#     if(is_finished(board)):
#         return
#     for i in get_available(board):
#         new_board, new_color = next_turn(np.copy(board), i, color)
#         reset_s(new_board, new_color)


