import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/Pendulum-v0
        Range action: [-2, 2]
        Range observation (tuple):
            - cos(theta): [-1, 1]
            - sin(theta): [-1, 1]
            - theta dot: [-8, 8]
        """
        return [np.random.uniform(-2, 2)]

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn. (Build model to approximate Q(s, a))
        """
        pass

class QLearningAgent:
    def __init__(self):
        """Init a new agent.
        # https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/README.md
        # https://stats.stackexchange.com/questions/184657/what-is-the-difference-between-off-policy-and-on-policy-learning
        """
        self.maxIteration = 1000
        self.gamma = 0.9 # discount rate parameter
        self.alpha = 0.5 # step-size parameter
        self.nbrActSamples = 100 # how many to sample at each time step
        self.nextActions = np.random.uniform(-2,2,self.nbrActSamples) # re-sampled each call
        self.Q = 0
        self.difference = 0
        self.weights = np.ones((4,)) # shape of range observation + action tuple
        self.epsilon = 0.1 # epsilon for epsilon greedy policy

    def act(self, observation):
        """
        - sample possible actions in [-2,2]
        - epsilon greedy policy to select next action (choose with best Q)
        """
        greedySample = np.random.uniform(0,1)
        if greedySample < self.epsilon:
            return([np.random.uniform(-2,2)]) # exploration
        else:
        # return best action according to Q(s,a) values
        # to estimate Q(s,a) use our estimator (weighted sum of features)

            # find best sampled action
            maxPossibleQ = 0
            bestAction = self.nextActions[0]
            for possibleAction in self.nextActions:
                # print(possibleAction)
                possibleQ = self.predict(observation,possibleAction)
                if possibleQ >= maxPossibleQ:
                    maxPossibleQ = possibleQ
                    # print("maxPossibleQ update:",maxPossibleQ)
                    bestAction = possibleAction

            # re-sample next actions for next function call
            self.nextActions = np.random.uniform(-2,2,self.nbrActSamples)
            bestActionArray = [bestAction]
            return(bestActionArray)  # sometimes array, sometimes not, WTF
            # return(bestAction)

    def reward(self, observation, action, reward):
        """
        Receive a reward for performing given action on
        given observation.

        This is where your agent can learn. (Build model to approximate Q(s, a))

        - features are 3 elements of observation tuple + action value
        """
        # action = action[0]
        # print("action in reward:",action)
        greedyQ = self.predict(observation,action)
        features = self.computeFeatures(observation,action)
        self.difference = (reward+self.gamma*greedyQ)-self.Q

        # we need "exact Q" below to update our weights
        self.Q += self.alpha*self.difference

        newQ = 0
        for wIndex in range(self.weights.shape[0]):
            self.weights[wIndex]+=self.alpha*self.difference*features[wIndex]
            newQ += self.weights[wIndex]*features[wIndex]
        self.Q = newQ

    def predict(self,observation,action):
        """
        predict Q(s,a)
        """
        features = self.computeFeatures(observation,action)
        estimQ = np.sum(np.multiply(self.weights,features))
        # print("estimQ:",estimQ)
        return(estimQ)

    def computeFeatures(self,observation,action):
        """
        - return array of features
        - this array should have the same size as the array of weights
        """
        # lets try features = observation and action
        featuresSize = len(observation) + 1 # 1 is so-called len(action)
        features = np.zeros((featuresSize,))
        for featureIndex in range(featuresSize):
            if (featureIndex < featuresSize-1):
                features[featureIndex] = observation[featureIndex]
            if (featureIndex == featuresSize-1):
                # sometimes array, sometimes not, WTF
                if isinstance(action,(list,)):
                    features[featureIndex] = action[0]
                #print("Action for features:",action)
                else:
                    features[featureIndex] = action
        # print("features returned:",features)
        return(features)

Agent = QLearningAgent
