# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent

DEFAULT_DISTANCE = 10000


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentFoodCount = currentGameState.getFood().count()

        # If agent does not eat food
        if len(newFood.asList()) == currentFoodCount:
            minDistance = DEFAULT_DISTANCE
            for pt in newFood.asList():
                minDistance = min(manhattanDistance(pt, newPos), minDistance)
        else:
            minDistance = 0

        for ghost in newGhostStates:  # the impact of ghost surges as distance get close
            minDistance += math.pow(2, (2 - manhattanDistance(ghost.getPosition(), newPos)))

        return -minDistance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        if not actions:
            return
        minimaxList = []
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            minimaxList.append((action, self.getMinActionValue(1, successor, self.depth)))
        return max(minimaxList, key=lambda res: res[1])[0]

    def getMaxActionValue(self, gameState, ply):
        result = []
        actions = gameState.getLegalActions(0)
        if not actions:
            return self.evaluationFunction(gameState)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            if successor.isWin() or successor.isLose():
                result.append(self.evaluationFunction(successor))
            else:
                result.append(self.getMinActionValue(1, successor, ply - 1))
        return max(result)

    def getMinActionValue(self, ghostIndex, gameState, ply):
        ghosts = gameState.getNumAgents() - 1
        result = []
        actions = gameState.getLegalActions(ghostIndex)
        if not actions:
            return self.evaluationFunction(gameState)
        for action in actions:
            successor = gameState.generateSuccessor(ghostIndex, action)
            if successor.isWin() or successor.isLose():
                result.append(self.evaluationFunction(successor))
            else:
                if ghostIndex == ghosts and ply == 1:
                    result.append(self.evaluationFunction(successor))
                elif ghostIndex == ghosts and ply != 1:
                    result.append(self.getMaxActionValue(successor, ply))
                else:
                    result.append(self.getMinActionValue(ghostIndex + 1, successor, ply))
        return min(result)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getMaxActionValue(gameState, 0, -math.inf, math.inf)[0]

    def getMaxActionValue(self, gameState, ply, alpha, beta):
        actions = gameState.getLegalActions(0)
        if not actions or gameState.isWin() or gameState.isLose() or ply == self.depth:
            return "", self.evaluationFunction(gameState)

        result = []
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.getMinActionValue(1, successor, ply, alpha, beta)
            result.append((action, value))
            maxVal = max(result, key=lambda res: res[1])[1]
            alpha = max(maxVal, alpha)
            if alpha > beta:
                break
        return max(result, key=lambda res: res[1])

    def getMinActionValue(self, ghostIndex, gameState, ply, alpha, beta):
        actions = gameState.getLegalActions(ghostIndex)
        if not actions or gameState.isWin() or gameState.isLose() or ply == self.depth:
            return self.evaluationFunction(gameState)
        result = []
        ghostCount = gameState.getNumAgents() - 1
        minVal = math.inf
        for action in actions:
            successor = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex != ghostCount:
                value = self.getMinActionValue(ghostIndex + 1, successor, ply, alpha, beta)
            else:
                x, value = self.getMaxActionValue(successor, ply + 1, alpha, beta)
            result.append(value)
            minVal = min(result)
            beta = min(minVal, beta)
            if beta < alpha:
                break
        return minVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getMaxActionValue(gameState, 0)[0]

    def getMaxActionValue(self, gameState, ply):
        actions = gameState.getLegalActions(0)
        if not actions or gameState.isLose() or gameState.isWin() or ply == self.depth:
            return "", self.evaluationFunction(gameState)
        result = []
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.getExpectiActionValue(successor, 1, ply)
            result.append((action, value))
        return max(result, key=lambda res: res[1])


    def getExpectiActionValue(self, gameState, ghostIndex, ply):
        ghostCount = gameState.getNumAgents() - 1
        actions = gameState.getLegalActions(ghostIndex)
        if not actions or gameState.isWin() or gameState.isLose() or ply == self.depth:
            return self.evaluationFunction(gameState)
        result = []
        for action in actions:
            successor = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex != ghostCount:
                value = self.getExpectiActionValue(successor, ghostIndex + 1, ply)
            else:
                x, value = self.getMaxActionValue(successor, ply + 1)
            result.append(value)
        return sum(result)/len(actions)



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return -10000
    elif currentGameState.isWin():
        return 10000
    pacmanPosition = currentGameState.getPacmanPosition()
    boundary = currentGameState.getWalls()
    food = currentGameState.getFood()
    boundaryCopy = boundary.copy()
    boundaryCopy[pacmanPosition[0]][pacmanPosition[1]] = 0
    result =0
    q = util.Queue()
    q.push(pacmanPosition)
    # BFS
    while True:
        x, y = q.pop()
        val = boundaryCopy[x][y] + 1
        if food[x][y]:
            break
        for v in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x = x + v[0]
            new_y = y + v[1]
            if not boundaryCopy[new_x][new_y]:
                boundaryCopy[new_x][new_y] = val
                q.push((new_x, new_y))
    result -= val

    ghosts = currentGameState.getGhostStates()
    for ghost in ghosts:
        #If ghost is scared, the agent gets a bonus
        if ghost.scaredTimer:
            result += 25
        else:
            result -= math.pow(100, (1.5 - manhattanDistance(ghost.getPosition(), pacmanPosition)))
    result -= 50 * food.count()
    return result

# Abbreviation
better = betterEvaluationFunction
