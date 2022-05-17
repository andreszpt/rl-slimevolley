import numpy as np
from matplotlib import pyplot as plt
from itertools import count, product


class Featurizer:
    def __init__(self, observation_space, centroids, sigma=0.5):
        self.n_dim = len(observation_space.low)
        self.centroids = np.array(centroids)     
        self.sigma = sigma
        self.n_parameters = len(self.centroids)
    def feature_vector(self, s):
        dist = (self.centroids - np.array(s))**2
        x = np.exp(-dist.sum(axis=1)/(2*self.sigma**2))
        return x
    
class ValueFunction:
    def __init__(self, featurizer, n_actions):
        self.n_actions = n_actions
        n_parameters = featurizer.n_parameters
        self.parameters = np.zeros((n_actions, n_parameters), dtype=np.float_) 
        self.featurizer = featurizer
        
    # estimar el valor de un estado y una acción dados
    def value(self, observation, action):
        features = self.featurizer.feature_vector(observation)
        return features.dot(self.parameters[action,:])

    # actualiza con state, action y target
    def update(self, observation, action, target, alpha):
        features = self.featurizer.feature_vector(observation)
        estimation = features.dot(self.parameters[action,:])
        delta = alpha * (target - estimation)*features
        self.parameters[action,:] = self.parameters[action,:] + delta
        
def get_action(observation, q_value_function, epsilon = 0.0):
    n_actions = q_value_function.n_actions
    if np.random.binomial(1, epsilon) == 1:
        return np.random.randint(n_actions)
    values = []
    for action in range(n_actions):
        values.append(q_value_function.value(observation, action))
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

def semi_gradient_sarsa(env, q, episodes, gamma = 1.0, epsilon = 0.1, epsilon_rate = 0.995, epsilon_min = 0.01,
                        alpha = 0.1, alpha_rate = 0.995, alpha_min = 0.01):
    history = np.zeros(episodes)
    history_average = np.zeros(episodes)
    for episode in range(episodes):
        if (episode+1) % 100 == 0 and episode > 0:
            mean_G = history_average[episode-1]
            print('episodio {}: alfa = {}, epsilon = {}, retorno medio = {}'.format(episode+1, alpha, epsilon, mean_G))
        G = 0 # retorno inicial del episodio
        S = env.reset()
        A = get_action(S, q, epsilon) # accion dada por epsilon greedy respecto a q
        for t in count():
            # if epsilon > epsilon_min:
            #     epsilon *= epsilon_rate
            # if alpha > alpha_min:
            #     alpha *= alpha_rate
            S_next, reward, done, _ = env.step(A)
            G += reward
            A_next = get_action(S_next, q, epsilon)
            if done:
                target = reward
                q.update(S, A, target, alpha)
                break 
            target = reward + gamma * q.value(S_next, A_next)
            q.update(S, A, target, alpha)
            A = A_next
            S = S_next        
        history[episode] = G
        history_average[episode] = np.mean(history[0:episode+1])        
    return q, history, history_average


def semi_gradient_n_step_sarsa(env, q, episodes, n=1, gamma = 1.0, epsilon = 0.1, alpha = 0.1):
    history = np.zeros(episodes)
    history_average = np.zeros(episodes)
    for episode in range(episodes):
        if (episode+1) % 100 == 0 and episode > 0:
            mean_G = history_average[episode-1]
            print('episodio {}: alfa = {}, epsilon = {}, retorno medio = {}'.format(episode+1, alpha, epsilon, mean_G))
        G = 0
        S = env.reset()
        A = get_action(S, q, epsilon)
        observations, actions, rewards = {}, {}, {}
        observations[0] = S
        actions[0] = A
        rewards[0] = 0.0
        T = float('inf') 
        for t in count():
            if t < T:
                S_next, R, done, _ = env.step(A)
                observations[t+1] = S_next
                rewards[t+1] = R
                if done:
                    T = t+1
                else:
                    A_next = get_action(S_next, q, epsilon)
                    actions[t+1] = A_next
                G += R
            tau = t - n + 1
            if tau >= 0:
                target = 0.0
                for i in range(tau+1, min(tau+n,T)):
                    target += gamma**(i-tau-1) * rewards[i]
                if tau + n < T:
                    target += gamma**n * q.value(observations[tau + n], actions[tau + n])
                q.update(observations[tau], actions[tau], target, alpha)
                del observations[tau]
                del actions[tau]
                del rewards[tau]
            if tau == T-1:
                break
            A = A_next       
        history[episode] = G
        history_average[episode] = np.mean(history[0:episode+1])        
    return q, history, history_average

def plot_history(history, history_average):
    ax = plt.subplot(111)
    plt.plot(history, label="retorno del episodio")
    plt.plot(history_average, label="retorno medio por episodio")
    plt.ylabel('Retorno completo', size=10)
    ax.legend(loc='lower right')
    plt.grid()
    plt.show()