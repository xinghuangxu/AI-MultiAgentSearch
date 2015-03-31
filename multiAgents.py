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
from random import randint
import random, util

from game import Agent
from pacman import GameState

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
       
        minDistance=99999
        if(len(newFood.asList())==0):
            minDistance=0
        for food in newFood.asList():
            newDistance= abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
            if(newDistance<minDistance):
                minDistance=newDistance
#         print "Min Food Distance: ",minDistance
#         print "New Food: ", newFood.asList()
        minGhostDistance=1000
        for x in newGhostStates:
            ghostPos=x.getPosition()
            newDistance=abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])
            if(2>newDistance):
                minGhostDistance= -100*(2-newDistance)
#         print "Min Ghost Distance: ",minGhostDistance
        "*** YOUR CODE HERE ***"
        return -minDistance+10*minGhostDistance

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
#     def getAction(self, gameState):
#         def max_value(state, currentDepth):
#             currentDepth = currentDepth + 1
#             if state.isWin() or state.isLose() or currentDepth == self.depth:
#                 return self.evaluationFunction(state)
#             v = float('-Inf')
#             for pAction in state.getLegalActions(0):
#                 self.sucessorCount+=1
#                 v = max(v, min_value(state.generateSuccessor(0, pAction), currentDepth, 1))
#             return v
#      
#         def min_value(state, currentDepth, ghostNum):
#             if state.isWin() or state.isLose():
#                 return self.evaluationFunction(state)
#             v = float('Inf')
#             for pAction in state.getLegalActions(ghostNum):
#                 if ghostNum == gameState.getNumAgents() - 1:
#                     self.sucessorCount+=1
#                     v = min(v, max_value(state.generateSuccessor(ghostNum, pAction), currentDepth))
#                 else:
#                     self.sucessorCount+=1
#                     v = min(v, min_value(state.generateSuccessor(ghostNum, pAction), currentDepth, ghostNum + 1))
#             return v
#         self.sucessorCount=0
#         # Body of minimax_decision starts here: #
#         pacmanActions = gameState.getLegalActions(0)
#         maximum = float('-Inf')
#         maxAction = ''
#         for action in pacmanActions:
#             currentDepth = 0
#             self.sucessorCount+=1
#             currentMax = min_value(gameState.generateSuccessor(0, action), currentDepth, 1)
#             if currentMax > maximum:
#                 maximum = currentMax
#                 maxAction = action
#         print maxAction,",", maximum
#         print "Succesor Called:",self.sucessorCount
#         return maxAction
    def getAction(self, gameState):
        self.sucessorCount=0;
#         print "Depth: ",treeDepth
        bestMove=None
        maxValue=-999999
        currentDepth = 0
        for action in gameState.getLegalActions(0):
            successorState=gameState.generateSuccessor(0, action)
            self.sucessorCount+=1
            value=self.minValue(successorState,currentDepth,1)
            if(value>maxValue):
                maxValue=value
                bestMove=action
        print bestMove,",", maxValue
#         print "Succesor Called:",self.sucessorCount
        return bestMove
      
      
    def maxValue(self,gameState,depth):
        depth+=1
        if depth==self.depth or gameState.isWin() or gameState.isLose(): #root
            return self.evaluationFunction(gameState)
        v=float('-Inf')
        agentIndex=0
        legalAcitons= gameState.getLegalActions(agentIndex)
        for action in legalAcitons:
            successorState=gameState.generateSuccessor(agentIndex, action)
            newValue=self.minValue(successorState,depth,1)
            if(v<newValue):
                v=newValue
        return v;
      
    def minValue(self,gameState,depth,ghostNum):
        if  gameState.isWin() or gameState.isLose(): #root
            return self.evaluationFunction(gameState)
        v=99999999
        legalAcitons= gameState.getLegalActions(ghostNum)
        for action in legalAcitons:
            successorState=gameState.generateSuccessor(ghostNum, action)
            if ghostNum == gameState.getNumAgents() - 1:
                newValue=self.maxValue(successorState,depth)
            else:
                newValue=self.minValue(successorState,depth,ghostNum+1)
            if(v>newValue):
                v=newValue
        return v;
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        
        bestMove=None
        a=float('-Inf')
        b=float('Inf')
        currentDepth = 0
        for action in gameState.getLegalActions(0):
            successorState=gameState.generateSuccessor(0, action)
            value=self.minValue(successorState,currentDepth,1,a,b)
            if(value>a):
                a=value
                bestMove=action
        return bestMove
      
      
    def maxValue(self,gameState,depth,a,b):
        depth+=1
        if depth==self.depth or gameState.isWin() or gameState.isLose(): #root
            return self.evaluationFunction(gameState)
        v=float('-Inf')
        agentIndex=0
        legalAcitons= gameState.getLegalActions(agentIndex)
        for action in legalAcitons:
            successorState=gameState.generateSuccessor(agentIndex, action)
            newValue=self.minValue(successorState,depth,1,a,b)
            if(v<newValue):
                v=newValue
            if(v>b): return v
            a=max(a,v)
        return v;
      
    def minValue(self,gameState,depth,ghostNum,a,b):
        if  gameState.isWin() or gameState.isLose(): #root
            return self.evaluationFunction(gameState)
        v=float('Inf')
        legalAcitons= gameState.getLegalActions(ghostNum)
        for action in legalAcitons:
            successorState=gameState.generateSuccessor(ghostNum, action)
            if ghostNum == gameState.getNumAgents() - 1:
                newValue=self.maxValue(successorState,depth,a,b)
            else:
                newValue=self.minValue(successorState,depth,ghostNum+1,a,b)
            if(v>newValue):
                v=newValue
            if(v<a): return v
            b=min(b,v)
        return v;

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        self.sucessorCount=0;
#         print "Depth: ",treeDepth
        bestMove=None
        maxValue=-999999
        currentDepth = 0
        for action in gameState.getLegalActions(0):
            successorState=gameState.generateSuccessor(0, action)
            self.sucessorCount+=1
            value=self.expectValue(successorState,currentDepth,1)
            if(value>maxValue):
                maxValue=value
                bestMove=action
        return bestMove
      
      
    def maxValue(self,gameState,depth):
        depth+=1
        if depth==self.depth or gameState.isWin() or gameState.isLose(): #root
            return self.evaluationFunction(gameState)
        v=float('-Inf')
        agentIndex=0
        legalAcitons= gameState.getLegalActions(agentIndex)
        for action in legalAcitons:
            successorState=gameState.generateSuccessor(agentIndex, action)
            newValue=self.expectValue(successorState,depth,1)
            if(v<newValue):
                v=newValue
        return v;
      
    def expectValue(self,gameState,depth,ghostNum):
        if  gameState.isWin() or gameState.isLose(): #root
            return self.evaluationFunction(gameState)
        legalAcitons= gameState.getLegalActions(ghostNum)
        total=0.0
        length=len(legalAcitons)
        for action in legalAcitons:
            successorState=gameState.generateSuccessor(ghostNum, action)
            if ghostNum == gameState.getNumAgents() - 1:
                newValue=self.maxValue(successorState,depth)
            else:
                newValue=self.expectValue(successorState,depth,ghostNum+1)
            total+=newValue
#         print total/float(length)
        return total/length;

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isLose(): 
        return -float("inf")
    elif currentGameState.isWin():
        return float("inf")
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    numberOfCapsulesLeft = len(currentGameState.getCapsules())
#     foods=currentGameState.food
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    total=0
    if randint(2,7) == 3: total-=200
    minDistance=99999
    numFood=len(newFood.asList()) 
    if(numFood==0):
        minDistance=0
    for food in newFood.asList():
        newDistance= abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
        if(newDistance<minDistance):
            minDistance=newDistance
    
     # ghost distance

  # active ghosts are ghosts that aren't scared.
    scaredGhosts, activeGhosts = [], []
    for ghost in currentGameState.getGhostStates():
        if not ghost.scaredTimer:
            activeGhosts.append(ghost)
        else: 
            scaredGhosts.append(ghost)

    def getManhattanDistances(ghosts): 
        return map(lambda g: util.manhattanDistance(newPos, g.getPosition()), ghosts)

    distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0

    if activeGhosts:
        distanceToClosestActiveGhost = min(getManhattanDistances(activeGhosts))
    else: 
        distanceToClosestActiveGhost = float("inf")
    distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)
    
    if scaredGhosts:
        distanceToClosestScaredGhost = min(getManhattanDistances(scaredGhosts))
    else:
        distanceToClosestScaredGhost = 0 # I don't want it to count if there aren't any scared ghosts

#     print foods
#     if((newPos[0]*100+newPos[1]) in foods):total+=100
#         print "Min Food Distance: ",minDistance
#         print "New Food: ", newFood.asList()
    minGhostDistance=1000
    for x in newGhostStates:
        if not x.scaredTimer:
            ghostPos=x.getPosition()
            newDistance=abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])
            if(2>newDistance):
                minGhostDistance= -100*(2-newDistance)
#         print "Min Ghost Distance: ",minGhostDistance
        "*** YOUR CODE HERE ***"
    total+=-1.5*minDistance+10*minGhostDistance-100*numFood-200*numberOfCapsulesLeft-10*distanceToClosestScaredGhost
#     print "Eval Info",currentGameState.getEvalInfo()
#     print "Number of Food: ",numFood
#     print "currentPosition:",newPos
#     print "Min Distance: ",minDistance
#     print "Has Food:",newFood[newPos[0]][newPos[1]]
#     print "Min Goast Distance:",minGhostDistance
#     print "food:",newFood.asList()
#     print "Total Score:",total
    return total


# Abbreviation
better = betterEvaluationFunction

