# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion

from .utils import *

directions = [(0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)]

def search(input: dict[tuple, tuple]) -> list[tuple]:
    """
    This is the entry point for your submission. The input is a dictionary
    of board cell states, where the keys are tuples of (r, q) coordinates, and
    the values are tuples of (p, k) cell states. The output should be a list of 
    actions, where each action is a tuple of (r, q, dr, dq) coordinates.

    See the specification document for more details.
    """

    # The render_board function is useful for debugging -- it will print out a 
    # board state in a human-readable format. Try changing the ansi argument 
    # to True to see a colour-coded version (if your terminal supports it).
    """ print(render_board(input, ansi=False))
    spread(input,(5,6),(-1,1))
    print(render_board(input, ansi=False))
    spread(input,(3,1),(0,1))
    print(render_board(input, ansi=False))
    spread(input,(3,2),(-1,1))
    print(render_board(input, ansi=False))
    spread(input,(1,4),(0,-1))
    print(render_board(input, ansi=False))
    spread(input,(1,3),(0,-1))
    print(render_board(input, ansi=False))  """
    """ print(render_board(input, ansi=False))
    spread(input, (6, 3), (-1, 0) )
    print(render_board(input, ansi=False)) """
    # Code above are all for testing
    print(render_board(input, ansi=False))
    actions = []
    if redWin(input):
        return actions
    while not redWin(input):
        path = aStarSearch(input, distance)
        #print(path, "path")
        spreadToken = path[0]
        spreadDestination = path[1]
        direction = getDirection(spreadToken, spreadDestination)
        action = spreadToken + direction
        actions.append(action)
        spread(input, spreadToken, direction)
        #print(action)
        print(render_board(input, ansi=False))
    print(actions)

    # Here we're returning "hardcoded" actions for the given test.csv file.
    # Of course, you'll need to replace this with an actual solution...
    return actions
