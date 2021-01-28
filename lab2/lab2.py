# MIT 6.034 Lab 2: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################


def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    return is_winning_chain(board) or board.count_pieces() == 42

def is_winning_chain(board):
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    if is_game_over_connectfour(board):
        return []

    return [board.add_piece(col) for col in range(7) if not board.is_column_full(col) ]

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    # -1000 if the minimizer has won, or 0 in case of a tie."""
    if not is_winning_chain(board):
        return 0
    
    if is_current_player_maximizer:
        return -1000
    else:
        return 1000

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner.""" 
    val = endgame_score_connectfour(board, is_current_player_maximizer)

    if not is_winning_chain(board) or val == 0:
        return 0
    elif val < 0: 
        return -1000 - (42 - board.count_pieces(True))
    else:
        return 1000 + (42 - board.count_pieces(False))

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    maximizer = get_heuristic_score(board, is_current_player_maximizer)
    minimizer = get_heuristic_score(board, not is_current_player_maximizer)
    return maximizer - minimizer

def get_heuristic_score(board, is_current_player_maximizer):
    score = 0

    scoring_system = {1: 1, 2: 10, 3: 100, 4: 500}
    for chain in board.get_all_chains(is_current_player_maximizer):
        chain_len = len(chain)
        score += scoring_system[chain_len]
    
    return score

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    best_path = []
    leaf_score = 0
    num_static_evals = 0

    all_paths = []
    all_paths.append([state])

    while len(all_paths) != 0:
        current_path = all_paths.pop(0)
        paths = get_paths(current_path)

        if not paths:
            num_static_evals += 1
            score = current_path[-1].get_endgame_score(True)
            if score >= leaf_score:
                leaf_score = score
                best_path = current_path

        all_paths = paths + all_paths

    return (best_path, leaf_score, num_static_evals)

def get_paths(path):
    paths = []

    children = path[-1].generate_next_states()
    for child in children:
        p = path.copy()
        p.append(child)
        if child not in path:
            paths.append(p)

    return paths

# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    best_path = [state]
    leaf_score = 0
    num_static_evals = 0

    if state.is_game_over():
        return (best_path, state.get_endgame_score(maximize), 1)
    
    all_tups = []
    for child in state.generate_next_states():
        all_tups.append(minimax_endgame_search(child, not maximize))

    num_static_evals = sum([tup[2] for tup in all_tups])
    sorted_tups = sorted(all_tups, key=lambda tup: tup[1])
    
    if maximize:
        best_path += sorted_tups[-1][0] 
        leaf_score = sorted_tups[-1][1]
    if not maximize:
        best_path += sorted_tups[0][0] 
        leaf_score = sorted_tups[0][1]
    
    return (best_path, leaf_score, num_static_evals)

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    best_path = [state]
    leaf_score = 0
    num_static_evals = 0
    
    if state.is_game_over():
        return (best_path, state.get_endgame_score(maximize), 1)

    if depth_limit == 0:
        return (best_path, heuristic_fn(state.get_snapshot(), maximize), 1)
    
    all_tups = []
    for child in state.generate_next_states():
        all_tups.append(minimax_search(child, heuristic_fn, depth_limit - 1, not maximize))

    num_static_evals = sum([tup[2] for tup in all_tups])
    sorted_tups = sorted(all_tups, key=lambda tup: tup[1])
    
    if maximize:
        best_path += sorted_tups[-1][0] 
        leaf_score = sorted_tups[-1][1]
    if not maximize:
        best_path += sorted_tups[0][0] 
        leaf_score = sorted_tups[0][1]
    
    return (best_path, leaf_score, num_static_evals)


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    best_path = [state]
    leaf_score = 0
    num_static_evals = 0
    
    if state.is_game_over():
        return (best_path, state.get_endgame_score(maximize), 1)

    if depth_limit == 0:
        return (best_path, heuristic_fn(state.get_snapshot(), maximize), 1)
    
    best_alpha = alpha
    best_beta = beta
    all_tups = []
    best_tup = None

    if maximize:    
        for child in state.generate_next_states():
            tup = minimax_search_alphabeta(child, best_alpha, beta, heuristic_fn, depth_limit - 1, not maximize)
            all_tups.append(tup)
            current_alpha = best_alpha
            best_alpha = max(best_alpha, tup[1])
            
            if best_alpha != current_alpha:
                best_tup = tup
            if best_alpha >= beta:
                break
        
        best_tup = max(all_tups, key = lambda tup: tup[1])
    
    if not maximize:
        for child in state.generate_next_states():
            tup = minimax_search_alphabeta(child, alpha, best_beta, heuristic_fn, depth_limit - 1, not maximize)
            all_tups.append(tup)
            current_beta = best_beta
            best_beta = min(best_beta, tup[1])
            
            if best_beta != current_beta:
                best_tup = tup
            if best_beta <= alpha:
                break
            
        best_tup = min(all_tups, key = lambda tup: tup[1])

    best_path += best_tup[0]
    leaf_score = best_tup[1]
    num_static_evals = sum([tup[2] for tup in all_tups])

    return (best_path, leaf_score, num_static_evals)


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()
    
    for depth in range(1, depth_limit + 1):
        anytime_value.set_value(minimax_search_alphabeta(
            state, -INF, INF, heuristic_fn, depth, maximize))
    
    return anytime_value


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Peyton Wang"
COLLABORATORS = "Marisa Papagelis"
HOW_MANY_HOURS_THIS_LAB_TOOK = 15
WHAT_I_FOUND_INTERESTING = "N/A"
WHAT_I_FOUND_BORING = "N/A"
SUGGESTIONS = "N/A"
