# rl-slimevolley
This repository is part of the Telematics Engineering Master's thesis titled "Evaluation of reinforcement learning algorithms in a multi-agent environment". The work focuses on evaluating a range of RL algorithms in the SlimeVolleyGym environment (https://github.com/hardmaru/slimevolleygym). Our experiments are based on the work of the original author of the environment.

The evaluated algorithms have different levels of complexity. Mainly, we classify them according to the value function or policy approximation that is used. Those that make a linear approximation are what we name classic, and their implementation is detailed in the project. On the other hand, those that make use of a non-linear approximation are considered advanced, and we will make use of the implementation given by the stable-baselines3 library. Furthermore, for this last type of algorithms, we will use the technique called self-play, which is based on the idea of learning by facing the agent against a previous version of itself.

The project was developed during the summer semester of 2022 in the Technical University of Cartagena, directed by Juan José Alcaraz Espín.

A `requirements.txt` file is provided so that the work environment can be easily replicated.
