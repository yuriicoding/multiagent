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
from functools import reduce

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        return_val = successorGameState.getScore()

        food_list = newFood.asList()

        if len(food_list) == 0:
            min_food = 0
        else:
            min_food = 100000

        for item in food_list:
            distance = manhattanDistance(newPos, item)
            if distance < min_food:
                min_food = distance

        min_ghost = 100000
        for item in newGhostStates:
            ghost_pos = item.configuration.pos
            distance = manhattanDistance(newPos, ghost_pos)
            if distance < min_ghost:
                min_ghost = distance

        if min_food == 0:
            min_food = 1
        if min_ghost == 1:
           return -100000
        return_val = return_val + min_ghost/(min_food*10)
        return return_val

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        def new_turn (playerIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            else:
                if playerIndex > 0:
                    return ghost_move(playerIndex, depth, gameState)
                else:
                    return pacman_move(playerIndex, depth, gameState)

        def ghost_move (playerIndex, depth, gameState):
            best_move = 100000
            all_players = gameState.getNumAgents()
            if playerIndex != all_players - 1:
                possible_actions = gameState.getLegalActions(playerIndex)
                successors = []
                for item in possible_actions:
                    successors += [gameState.generateSuccessor(playerIndex, item)]
                for item in successors:
                    move = new_turn(playerIndex + 1, depth, item)
                    best_move = min(best_move, move)
                return best_move
            else:
                possible_actions = gameState.getLegalActions(playerIndex)
                successors = []
                for item in possible_actions:
                    successors += [gameState.generateSuccessor(playerIndex, item)]
                for item in successors:
                    move = new_turn(0, depth - 1, item)
                    best_move = min(best_move, move)
                return best_move

        def pacman_move (playerIndex, depth, gameState):
            best_move = -100000
            possible_actions = gameState.getLegalActions(playerIndex)
            successors = []
            for item in possible_actions:
                successors += [gameState.generateSuccessor(playerIndex, item)]
            for item in successors:
                move = new_turn(playerIndex + 1, depth, item)
                best_move = max(best_move, move)
            return best_move

        def choose_action (gameState, depth):
            successor_list = []
            for item in gameState.getLegalActions(0):
                successor_list += [(gameState.generateSuccessor(0, item), item)]
            node_value = -100000
            best_action = None
            for successor in successor_list:
                new_value = new_turn(1, depth, successor[0])
                if new_value > node_value:
                    node_value = new_value
                    best_action = successor[1]
            return best_action


        return choose_action(gameState, self.depth)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def new_turn(playerIndex, depth, gameState, max_check, min_check):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            else:
                if playerIndex > 0:
                    return ghost_move(playerIndex, depth, gameState, max_check, min_check)
                else:
                    return pacman_move(playerIndex, depth, gameState, max_check, min_check)

        def ghost_move(playerIndex, depth, gameState, max_check, min_check):
            best_move = 100000
            all_players = gameState.getNumAgents()
            if playerIndex != all_players - 1:
                possible_actions = gameState.getLegalActions(playerIndex)
                for item in possible_actions:
                    successor = gameState.generateSuccessor(playerIndex, item)
                    move = new_turn(playerIndex + 1, depth, successor, max_check, min_check)
                    best_move = min(best_move, move)
                    if best_move < max_check:
                        return best_move
                    min_check = min(best_move, min_check)
                return best_move
            else:
                possible_actions = gameState.getLegalActions(playerIndex)
                for item in possible_actions:
                    successor = gameState.generateSuccessor(playerIndex, item)
                    move = new_turn(0, depth - 1, successor, max_check, min_check)
                    best_move = min(best_move, move)
                    if best_move < max_check:
                        return best_move
                    min_check = min(best_move, min_check)
                return best_move

        def pacman_move(playerIndex, depth, gameState, max_check, min_check):
            best_move = -100000
            possible_actions = gameState.getLegalActions(playerIndex)
            for item in possible_actions:
                successor = gameState.generateSuccessor(playerIndex, item)
                move = new_turn(playerIndex + 1, depth, successor, max_check, min_check)
                best_move = max(best_move, move)
                if best_move > min_check:
                    return best_move
                max_check = max(best_move, max_check)
            return best_move

        max_check = -100000
        min_check = 100000
        possible_actions = gameState.getLegalActions(0)
        best_move = -100000
        best_action = None

        for item in possible_actions:
            successor = gameState.generateSuccessor(0, item)
            shortcut = new_turn(1, self.depth, successor, max_check, min_check)
            if shortcut > best_move:
                best_move = shortcut
                best_action = item
            max_check = max(max_check, best_move)

        return best_action

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

        def new_turn(playerIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            else:
                if playerIndex > 0:
                    return ghost_move(playerIndex, depth, gameState)
                else:
                    return pacman_move(playerIndex, depth, gameState)

        def ghost_move(playerIndex, depth, gameState):
            best_move = 100000
            all_players = gameState.getNumAgents()
            if playerIndex != all_players - 1:
                possible_actions = gameState.getLegalActions(playerIndex)
                all_poss = len(possible_actions)
                successors = []
                for item in possible_actions:
                    successors += [gameState.generateSuccessor(playerIndex, item)]
                for item in successors:
                    move = new_turn(playerIndex + 1, depth, item)
                    best_move = best_move + move/all_poss
                return best_move
            else:
                possible_actions = gameState.getLegalActions(playerIndex)
                all_poss = len(possible_actions)
                successors = []
                for item in possible_actions:
                    successors += [gameState.generateSuccessor(playerIndex, item)]
                for item in successors:
                    move = new_turn(0, depth - 1, item)
                    best_move = best_move + move/all_poss
                return best_move

        def pacman_move(playerIndex, depth, gameState):
            best_move = -100000
            possible_actions = gameState.getLegalActions(playerIndex)
            successors = []
            for item in possible_actions:
                successors += [gameState.generateSuccessor(playerIndex, item)]
            for item in successors:
                move = new_turn(playerIndex + 1, depth, item)
                best_move = max(best_move, move)
            return best_move

        def choose_action(gameState, depth):
            successor_list = []
            for item in gameState.getLegalActions(0):
                successor_list += [(gameState.generateSuccessor(0, item), item)]
            node_value = -100000
            best_action = None
            for successor in successor_list:
                new_value = new_turn(1, depth, successor[0])
                if new_value > node_value:
                    node_value = new_value
                    best_action = successor[1]
            return best_action

        return choose_action(gameState, self.depth)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    food_list = food.asList()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = []
    for item in ghostStates:
        ghostPositions += [item.getPosition()]
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    totalScare = 0
    for item in scaredTimes:
        totalScare += item
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    food_dis = []
    for item in food_list:
        food_dis += [1.0/manhattanDistance(pos, item)]
    capsule_dis = []
    for item in capsules:
        capsule_dis += [1.0/manhattanDistance(pos, item)]
    ghost_dis = []
    for item in ghostPositions:
        ghost_dis += [manhattanDistance(pos, item)]

    average = lambda x: float(sum(x)) / len(x)

    if len(food_dis) == 0:
        food_average = float("inf")
    else:
        food_average = average (food_dis) * 10

    if len(capsule_dis) == 0:
        capsule_average = 0
    else:
        if totalScare == 0:
            capsule_average = average(capsule_dis) * 10
        else:
            capsule_average = average(capsule_dis) * 10 * (-1)

    if len(ghost_dis) == 0:
        min_ghost = 0
        ghostScore = min(ghost_dis) / 2.0 * (0.7 if sum(scaredTimes) > 0 else -1.5)
    else:
        if totalScare > 0:
            min_ghost = 0.7 * min(ghost_dis) / 5.0
        else:
            min_ghost = (-1.5) * min(ghost_dis) / 5.0

    ret_value = score + food_average + capsule_average + min_ghost

    return ret_value

# Abbreviation
better = betterEvaluationFunction
