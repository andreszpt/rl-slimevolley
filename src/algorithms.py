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
    
    
####################################################################
#                       SARSA-LAMBDA                               #
####################################################################
class ValueFunctionSL:
    def __init__(self, featurizer, n_actions):
        self.n_actions = n_actions
        self.n_parameters = featurizer.n_parameters
        self.featurizer = featurizer
        self.reset()
        
    def reset(self):
        self.parameters = np.zeros((self.n_actions, self.n_parameters), dtype=np.float_) 
        
    # estimar el valor de un estado y una acción dados
    def value(self, observation, action):
        features = self.featurizer.feature_vector(observation)
        return features.dot(self.parameters[action,:])

    # actualiza con state, action y target
    def update(self, observation, action, target, step_size):
        features = self.featurizer.feature_vector(observation)
        estimation = features.dot(self.parameters[action,:])
        delta = step_size * (target - estimation)*features
        self.parameters[action,:] = self.parameters[action,:] + delta
        
def get_action(observation, q_value_function, epsilon = 0.0):
    n_actions = q_value_function.n_actions
    if np.random.binomial(1, epsilon) == 1:
        return np.random.randint(n_actions)
    values = []
    for action in range(n_actions):
        values.append(q_value_function.value(observation, action))
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

class LambdaValueFunction(ValueFunctionSL):
    def __init__(self, featurizer, n_actions, L):
        super().__init__(featurizer, n_actions)
        self.L = L
        self.initialize_z()
    
    def initialize_z(self):
        self.z = np.zeros((self.n_actions, self.featurizer.n_parameters), dtype=np.float_)
        
    def update(self, observation, action, TD_error, alpha, gamma):
        # actualiza z
        features = self.featurizer.feature_vector(observation)
        self.z = gamma*self.L*self.z
        self.z[action, :] = self.z[action, :] + features
        
        # actualiza parametros
        self.parameters += alpha*TD_error*self.z

def sarsa_lambda(env, q, episodes, t_max, gamma = 1.0, epsilon = 0.1, alpha = 0.1):
    n_actions = env.action_space.n
    history = np.zeros(episodes)
    history_average = np.zeros(episodes)
    total_t = 0
    for episode in range(episodes):
        if (episode+1) % 2 == 0 and episode > 0:
            mean_G = history_average[episode-1]
            print('episodio {}: alfa = {}, epsilon = {}, retorno medio = {}'.format(episode+1, alpha, epsilon, mean_G))
        if total_t > t_max:
            break
        G = 0
        t = 0
        S = env.reset()
        A = get_action(S, q, epsilon)
        done = False
        q.initialize_z()
        while not done:
            if (total_t+1) % 50 == 0:
                print('t = {}'.format(total_t+1))
            epsilon *= 0.99999
            alpha *= 0.99999
            S_next, R, done, _ = env.step(A) # aplicamos la accion            
            Q = q.value(S,A) # obtenemos Q (lo necesitaremos para calcular TD_error)            
            if done:
                Q_next = 0.0
            else:
                A_next = get_action(S_next, q, epsilon)
                Q_next = q.value(S_next, A_next)            
            TD_error = R + gamma * Q_next - Q
            q.update(S,A,TD_error,alpha,gamma)           
            S = S_next
            A = A_next
            G += R # actualizamos G para las graficas
            t += 1 # contador de etapas     
            total_t += 1
        history[episode] = G
        history_average[episode] = np.mean(history[0:episode+1])        
    return q, history, history_average


####################################################################
#                    REINFORCE CON BASELINE                        #
####################################################################
class PolicyEstimator:
    def __init__(self, featurizer, n_actions):
        self.n_actions = n_actions
        n_parameters = featurizer.n_parameters
        # self.parameters = np.random.rand(n_actions, n_parameters)*0.1
        self.parameters = np.zeros((n_actions, n_parameters), dtype=np.float_) 
        self.featurizer = featurizer

    def __getitem__(self, S):
        return self.get_action(S)
    
    # devuelve la función de probabilidad dada una observación
    def get_pi_distribution(self, observation):
        features = self.featurizer.feature_vector(observation)
        h = np.zeros((self.n_actions,), dtype=np.float_)
        for a in range(self.n_actions):
            h[a] = features.dot(self.parameters[a,:])
        pi = np.exp(h) / np.sum(np.exp(h))
        if np.isnan(pi).any():
            print('¡Algo salió mal! El algoritmo puede ser divergente. Intenta con un alfa más pequeño')
        return pi
        
    # muestrea una acción usando la función de probabilidad para un estado dado (observation)
    def get_action(self, observation):
        pi = self.get_pi_distribution(observation)
        cdf = np.cumsum(pi)
        s = np.random.random()
        action = np.where(s < cdf)[0][0]
        return action

    # actualiza el vector parámetros dado el estado, la acción y el error
    def update(self, observation, action, error, alpha, discount = 1.0):
        features = self.featurizer.feature_vector(observation)
        pi = self.get_pi_distribution(observation)
        for a in range(self.n_actions):
            if a==action:
                grad_log = features - pi[a]*features
            else:
                grad_log = -1.0*pi[a]*features 
            delta = alpha * discount * error * grad_log
            self.parameters[a,:] = self.parameters[a,:] + delta

def generate_episode(env, policy, A = -1, max_t = 1000):
    state_action_seq, rewards = [], []
    S = env.reset()
    if A < 0:
        A = policy[S]
    state_action_seq.append((S, A))
    S, reward, done, _ = env.step(A)
    rewards.append(reward)
    t = 0
    while not done and t < max_t:
        A = policy[S]
        state_action_seq.append((S, A))
        S, reward, done, _ = env.step(A)
        rewards.append(reward)
        t += 1
    return state_action_seq, rewards

class ValueFunctionRB():
    def __init__(self, featurizer):
        n_parameters = featurizer.n_parameters
        # self.parameters = np.random.rand(1, n_parameters)*0.1
        self.parameters = np.zeros((n_parameters,), dtype=np.float_) 
        self.featurizer = featurizer
        
    # estima el valor de un estado dado (observation)
    def value(self, observation):
        features = self.featurizer.feature_vector(observation)
        return features.dot(self.parameters)

    # actualiza el vector de pesos w con el estado (observation) y el error dado
    def update(self, observation, error, alpha):
        features = self.featurizer.feature_vector(observation)
        delta = alpha * error * features
        self.parameters = self.parameters + delta
        
def reinforce_baseline(env, pi, v, episodes, t_max, alpha, beta, gamma=1.0):
    history = np.zeros(episodes)
    history_average = np.zeros(episodes)
    total_t = 0
    best_parameters = 0
    for episode in range(episodes):
        if episode % 2 == 0 and episode > 0:
            mean_G = history_average[episode-1]
            print('episodio {}: alpha = {}, beta = {}, retorno medio = {}'.format(episode, alpha, beta, mean_G))
        if total_t > t_max:
            break
        # genera el episodio y almacena los pares estado acción y las recompensas
        state_action_seq, rewards = generate_episode(env, pi)
        G = 0
        for t in range(len(state_action_seq) - 1, -1, -1):
            if (total_t+1) % 50 == 0:
                print('t = {}'.format(total_t+1))
            alpha *= 0.99999
            beta *= 0.99999
            R = rewards[t]
            (S,A) = state_action_seq[t] # extrae el par estado acción
            G = R + gamma * G # calcula retorno en t
            I = gamma ** t # variable de descuento
            error = G - v.value(S) # error de la estimación v en S respecto a G
            v.update(S, error, beta)  # actualiza v
            pi.update(S, A, error, alpha, I) # actualiza pi
            total_t += 1
        history[episode] = G
        history_average[episode] = np.mean(history[0:episode+1])
        if G >= history.max():
            best_parameters = np.copy(pi.parameters)
    return pi, history, history_average, best_parameters