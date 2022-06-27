import numpy as np
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

def sarsa_lambda(env, q, t_max, gamma = 1.0, epsilon = 0.1, alpha = 0.1):
    n_actions = env.action_space.n
    rewards = np.zeros((0))
    lengths = np.zeros((0))
    total_t = 0
    episode = 0
    while total_t < t_max:
        G = 0
        t = 0
        S = env.reset()
        A = get_action(S, q, epsilon)
        done = False
        q.initialize_z()
        while not done:
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
        total_t += t
        episode += 1
        print('episodio {}: alfa = {}, epsilon = {}, rew_ep = {}, len_ep = {}, total_t = {}'.format(episode, alpha, epsilon, G, t, total_t))
        rewards = np.append(rewards, G)
        lengths = np.append(lengths, t)
    return q, rewards, lengths


####################################################################
#                    ACTOR CRITIC + TRAZAS                         #
####################################################################
class ValueFunctionAC():
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


class LambdaValueFunctionAC(ValueFunctionAC):
    def __init__(self, featurizer, L):
        super().__init__(featurizer)
        self.L = L
        self.initialize_z()
    
    def initialize_z(self):
        self.z = np.zeros((self.featurizer.n_parameters), dtype=np.float_)
        
    def update(self, observation, error, alpha, gamma):
        # actualiza z
        features = self.featurizer.feature_vector(observation)
        self.z = gamma*self.L*self.z
        self.z = self.z + features
        
        # actualiza parametros
        self.parameters += alpha*error*self.z
        
class PolicyEstimatorAC:
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
        
class LambdaPolicyEstimatorAC(PolicyEstimatorAC):
    def __init__(self, featurizer, n_actions, L):
        super().__init__(featurizer, n_actions)
        self.L = L
        self.initialize_z()
    
    def initialize_z(self):
        self.z = np.zeros((self.n_actions, self.featurizer.n_parameters), dtype=np.float_)
        
    def update(self, observation, action, error, alpha, gamma, I):
        # actualiza z
        features = self.featurizer.feature_vector(observation)
        self.z = gamma*self.L*self.z
        pi = self.get_pi_distribution(observation)
        for a in range(self.n_actions):
            if a==action:
                grad_log = features - pi[a]*features
            else:
                grad_log = -1.0*pi[a]*features 
            self.z[a, :] = self.z[a, :] + I*grad_log
        
        # actualiza parametros
        self.parameters += alpha*error*self.z
        

def actor_critic_lambda(env, pi, v, t_max, alpha, beta, gamma=1.0, max_t = np.Inf):
    n_actions = env.action_space.n
    rewards = np.zeros((0))
    lengths = np.zeros((0))
    total_t = 0
    episode = 0
    while total_t < t_max:
        # Resetea el entorno y obtiene el primer estado visitado
        S = env.reset()    
        I = 1.0     
        done = False
        t = 0
        G = 0
        # inicializamos las trazas de elegibilidad
        pi.initialize_z()
        v.initialize_z()
        # Avanzamos una etapa del episodio en cada iteración
        while not done and t < max_t:
            alpha *= 0.99995
            beta *= 0.99995
            A = pi.get_action(S)
            S_next, R, done, _ = env.step(A)
            # Calcula el target TD y el error TD
            if done:
                v_next = 0.0
            else:
                v_next = v.value(S_next)
            td_error = R + gamma * v_next - v.value(S)
            # CRITIC: Actualiza el estimador de la función valor, v
            v.update(S, td_error, beta, gamma)
            # ACTOR: Actualiza el estimador de la política, pi, usando el error TD
            pi.update(S, A, td_error, alpha, gamma, I)
            # Actualiza G, I, S y el contador de etapas t
            G += I*R
            I = gamma * I
            S = S_next
            t += 1        
            # ################################################################## #        
        total_t += t
        episode += 1
        print('episodio {}: alpha = {}, beta = {}, rew_ep = {}, len_ep = {}, total_t = {} '.format(episode, alpha, beta, G, t, total_t))
        rewards = np.append(rewards, G)
        lengths = np.append(lengths, t)   
        if G >= rewards.max():
            best_parameters = np.copy(pi.parameters)
    return pi, rewards, lengths, best_parameters