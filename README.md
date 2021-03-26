# Reinforcement_Learning_DQN

This repository contains coursework which trains a RL agent to navigate randomly generated mazes.

Files _train_and_test_ and _random_environment_ have been provided. The _random_environment_ file generates mazes and the _train_and_test_ file trains the agent on one of such randomly generated mazes for 10 minutes and evaluates its performance.

The _agent_ file has the following methods implemented:
*   Double Q learning
*   Decaying epsilon-greedy policy
*   Target network
*   Early stopping of training
*   Prioritised experience replay
*   A reward fucntion that penalizes hitting a wall and provides high rewards in close proximity of the target

This is a coursework completed as a part of the ICL Reinforcement Learning module taught by Dr A. Faisal & Dr. E. Johns.
