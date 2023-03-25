# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion
from collections import defaultdict
from queue import PriorityQueue
import math

def apply_ansi(str, bold=True, color=None):
    """
    Wraps a string with ANSI control codes to enable basic terminal-based
    formatting on that string. Note: Not all terminals will be compatible!

    Arguments:

    str -- String to apply ANSI control codes to
    bold -- True if you want the text to be rendered bold
    color -- Colour of the text. Currently only red/"r" and blue/"b" are
        supported, but this can easily be extended if desired...

    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{str}\033[0m"

def render_board(board: dict[tuple, tuple], ansi=False) -> str:
    """
    Visualise the Infexion hex board via a multiline ASCII string.
    The layout corresponds to the axial coordinate system as described in the
    game specification document.
    
    Example:

        >>> board = {
        ...     (5, 6): ("r", 2),
        ...     (1, 0): ("b", 2),
        ...     (1, 1): ("b", 1),
        ...     (3, 2): ("b", 1),
        ...     (1, 3): ("b", 3),
        ... }
        >>> print_board(board, ansi=False)

                                ..     
                            ..      ..     
                        ..      ..      ..     
                    ..      ..      ..      ..     
                ..      ..      ..      ..      ..     
            b2      ..      b1      ..      ..      ..     
        ..      b1      ..      ..      ..      ..      ..     
            ..      ..      ..      ..      ..      r2     
                ..      b3      ..      ..      ..     
                    ..      ..      ..      ..     
                        ..      ..      ..     
                            ..      ..     
                                ..     
    """
    dim = 7
    output = ""
    for row in range(dim * 2 - 1):
        output += "    " * abs((dim - 1) - row)
        for col in range(dim - abs(row - (dim - 1))):
            # Map row, col to r, q
            r = max((dim - 1) - row, 0) + col
            q = max(row - (dim - 1), 0) + col
            if (r, q) in board:
                color, power = board[(r, q)]
                text = f"{color}{power}".center(4)
                if ansi:
                    output += apply_ansi(text, color=color, bold=False)
                else:
                    output += text
            else:
                output += " .. "
            output += "    "
        output += "\n"
    return output

def correctCoordinates(coordinates: tuple):
    """
    this function is being used to correct the corrdinates
    for example: (7, 7) -> (0, 0)
    """
    r = coordinates[0]
    q = coordinates[1]
    if r < 0:
        r = 7 - abs(r) % 7
    else:
        r = r % 7
    if q < 0:
        q = 7 - abs(q) % 7
    else:
        q = q % 7
    return (r, q)
    
def calculateBoardCost(board):
    cost = 0
    for token in board.keys():
        if board[token][0] == 'b':
            cost += 1
    return cost

def spread(board: dict[tuple, tuple], token: tuple, direction: tuple):
    """
    spread function. The input is the board status, tokens coordinates
    that about to move, and move direction
    """
    color = board[token][0]
    power = board[token][1]
    curr_tok = token
    
    while power > 0 :
        curr_tok = correctCoordinates((curr_tok[0] + direction[0], curr_tok[1] + direction[1]))
        addToken(board,curr_tok,color)
        power -= 1

    # delete the token being spreaded
    del board[token]
    

def addToken(board: dict[tuple, tuple], token: tuple, color: str):
    """
    Add a token to the board, increment its power if it's already present,
    and remove it if its power reaches 7.
    """
    if token in board:
        current_power = board[token][1] + 1
        if current_power < 7:
            board[token] = (color, current_power)
        else:
            del board[token]
    else:
        board[token] = (color, 1)

def distance(p1, p2):
    """
    eulidean distance of two points
    """
    #return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return 1

def manhattan_distance(p1, p2):
    """
    Manhattan distance of two points
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def findClosestBlueTokens(redToken, blueTokens):
    """
    this function find the closest blue token and red token
    """
    minDistance = 1000
    closestBlueToken = blueTokens[0]
    for blueToken in blueTokens:
        tokDistance = chebyshevDistance(redToken, blueToken)
        #print(redToken,blueToken,tokDistance)
        if tokDistance < minDistance:
            minDistance = tokDistance
            closestBlueToken = blueToken
    return closestBlueToken

def sortBoardByPower(board: dict[tuple, tuple]):
    """
    Sort the board dictionary by token power in descending order
    """
    sorted_board = sorted(board.items(), key=lambda x: x[1][1], reverse=True)
    return dict(sorted_board)

def findFarthestSpreadedNeighbours(token, power):
    """
    this funciton find all spreaded neigoubours
    """    
    directions = [(0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)]
    neighbours = []
    for direction in directions:
        neighbour = (token[0] + (direction[0] * power), token[1] + (direction[1] * power))
        neighbours.append(correctCoordinates(neighbour))
    return neighbours

def findAllSpreadNeighbours(token, power):
    directions = [(0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)]
    neighbours = []
    for direction in directions:
        #sameDirectionNeighbour = []
        powerCopy = power
        while powerCopy > 0:
            neighbour = correctCoordinates((token[0] + (direction[0] * powerCopy), token[1] + (direction[1] * powerCopy)))
            #sameDirectionNeighbour.append(correctCoordinates(neighbour))
            powerCopy -= 1
            neighbours.append(neighbour)
    return neighbours

def findAllNeighbours(token):
    """
    this funciton find all spreaded neigoubours
    """    
    directions = [(0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)]
    neighbours = []
    for direction in directions:
        neighbour = (token[0] + direction[0], token[1] + direction[1])
        neighbours.append(correctCoordinates(neighbour))
    return neighbours
    
def divideTokens(board: dict[tuple, tuple]):
    """
    divide blue tokens and red tokens
    """
    redTokens = []
    blueTokens = []
    sortboard = sortBoardByPower(board)
    # divide tokens by color
    for token in sortboard.keys():
        color = board[token][0]
        if color == 'r':
            redTokens.append(token)
        else:
            blueTokens.append(token)
    #print(redTokens," red")
    return (redTokens, blueTokens)

def redWin(board: dict[tuple, tuple]):
    """
    wining condition for red
    """
    tokens = set()
    for token in board.keys():
        color = board[token][0]
        tokens.add(color)

    if "b" in tokens:
        return False
    return True

def mapCoordinates(p1):
    if p1 == (6, 6):
        return [p1, (-1, -1), (-1, 6), (6, -1)]
    if p1 == (0, 0):
        return [p1, (7, 7), (7, 0), (0, 7)]
    if p1[0] == 6:
        return [p1, (-1, p1[1])]
    if p1[1] == 6:
        return [p1, (p1[0], -1)]
    if p1[0] == 0:
        return [p1, (7, p1[1])]
    if p1[1] == 0:
        return [p1, (p1[0], 7)]
    return [p1]
def getDirection(p1, p2, board):
    """
    calculates the moving direction given two points
    """
    directions = [(0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)]
    coordinatesDifference = set([-1, 0, 1])
    direction_r = p2[0] - p1[0]
    direction_q = p2[1] - p1[1]
    power = board[p1][1]
    if (abs(direction_r) in range(1, 6)) or (abs(direction_q) in range(1, 6)):
        for direction in directions:
            converted_r = p1[0] + (direction[0] * power)
            converted_q = p1[1] + (direction[1] * power)
            #print(converted_r, converted_q)
            if correctCoordinates((converted_r, converted_q)) == p2:
                return direction
            
    if direction_r == 0:
        if direction_q > 1 and direction_q < 6:
            direction_q = 1
        if direction_q > -6 and direction_q < -1:
            direction_q = -1
    if direction_q == 0:
        if direction_r > 1 and direction_r < 6:
            direction_r = 1
        if direction_r > -6 and direction_r < -1:
            direction_r = -1

    if direction_r not in coordinatesDifference:
        if direction_r == 6:
            direction_r = -1
        if direction_r == -6:
            direction_r = 1

    if direction_q not in coordinatesDifference:
        if direction_q == 6:
            direction_q = -1
        if direction_q == -6:
            direction_q = 1

    return (direction_r, direction_q)

def chebyshevDistance(p1, p2):
    """
    chebyshev distance
    """
    mappedCoordinates = mapCoordinates(p1)
    mappedDistance = []
    p1_r = p1[0]
    p1_q = p1[1]
    p2_r = p2[0]
    p2_q = p2[1]
    for cor in mappedCoordinates:
        cor_r = cor[0]
        cor_q = cor[1]
        mappedDistance.append(max(abs(p2_r - cor_r), abs(p2_q - cor_q), abs(cor_r + cor_q - p2_r - p2_q)))
    mappedChebyshev = min(mappedDistance)
    chebyshevDistance = max(abs(p2_r - p1_r), abs(p2_q - p1_q), abs(p1_r + p1_q - p2_r - p2_q))
    if mappedChebyshev < chebyshevDistance:
        return mappedChebyshev
    return chebyshevDistance

def aStarSearch(board: dict[tuple, tuple], heuristic):

    copyBoard = board.copy()
    # group redTokens and blueTokens
    dividedTokens = divideTokens(board)
    redTokens = dividedTokens[0]
    blueTokens = dividedTokens[1]

    # find closest two tokens
    #closestPair = findClosestTwoTokens(redTokens, blueTokens)
    #startToken = closestPair[0]
    #startPower = board[startToken][1]
    #endToken = closestPair[1] 

    #priorityQ = PriorityQueue()
    #priorityQ.put((0, startToken))
    #cameFrom = defaultdict(tuple)
    #cost = defaultdict(float)
    paths = []
    for redToken in redTokens:
        for blueToken in blueTokens:
            startToken = redToken
            startPower = board[startToken][1]
            endToken = blueToken
            #endToken = findClosestBlueTokens(startToken, blueTokens)
            #print(startToken, endToken)
            #allNeighbours = findAllSpreadNeighbours(startToken, startPower)
            #allNeighbours = findAllNeighbours(startToken, endToken)
            '''if endToken in allNeighbours:
                paths.append([startToken, endToken])
                continue'''
            tmpBoard = board.copy()
            priorityQ = PriorityQueue()
            priorityQ.put((0, startToken, tmpBoard))
            cameFrom = defaultdict(tuple)
            cost = defaultdict(float)
            
            
            while not priorityQ.empty():
                p, currentToken, b = priorityQ.get()
                #print(currentToken, "curr")
                if currentToken == endToken:
                    break
                if currentToken not in b.keys():
                    continue
                '''if currentToken == startToken:
                    neighbours = findAllSpreadNeighbours(startToken, startPower)
                    #neighbours = findAllNeighbours(startToken)
                else:
                    neighbours = findAllNeighbours(currentToken)'''
                #print(b)
                neighbours = findAllSpreadNeighbours(currentToken, b[currentToken][1])
                print(currentToken)
                print(neighbours)
                for neighbour in neighbours:
                    copyB = b.copy()
                    newCost = cost[currentToken] + 1
                    tmpDirection = getDirection(currentToken, neighbour, copyB)
                    spread(copyB, currentToken, tmpDirection)
                    if neighbour not in cost or newCost < cost[neighbour]:
                        cost[neighbour] = newCost
                        priority = newCost + heuristic(endToken, neighbour)
                        #print("current ", currentToken," goes into ", neighbour, " with h", heuristic(endToken, neighbour), "new cost:", newCost, "p = ", priority)
                        priorityQ.put((priority, neighbour, copyB))
                        cameFrom[neighbour] = currentToken

            path = [endToken]
            while path[-1] != startToken:
                path.append(cameFrom[path[-1]])
            path.reverse()
            paths.append(path)
    print(paths, "THIS IS PATHSSSS")

    currentCost = calculateBoardCost(board)
    costs = []
    for path in paths:
        boardCopy = board.copy()
        direction = getDirection(path[0], path[1], board)
        spread(boardCopy, path[0], direction)
        cost = calculateBoardCost(boardCopy)
        costs.append(cost)

    minCost = min(costs)
    minIndex = costs.index(minCost)
    if minCost < currentCost:
        return paths[minIndex]

    minPathIndex = 0
    minPathLength = 10000
    for i in range(len(paths)):
        pathLength = len(paths[i])
        if pathLength < minPathLength:
            minPathLength = pathLength
            minPathIndex = i

    return paths[minPathIndex]

