# Q_Learning_Flappy_Bird

## Description

Following is the code for the implementation of the following paper in which a simple Q-Learning algorithm was trained using positional values to play the Flappy Bird game: https://kilyos.ee.bilkent.edu.tr/~eee546/FlappyQ.pdf

The implementation follows two variations of the game, one with constant speed while falling and the other with constant gravity.

Following that, the same was implemented using Deep Q-Network for higher accuracy and better results, using a 3 layer Dense network.

## Code

The flappy.py and flappy_gravity.py files are used to train the Q-learning algorithm over a set number of episodes, while test.py is used to load a saved Q_table to view the results in the form of the game.

flappy_nn.py is similarly used to trained the DQN, while test_nn.py is used to load a model and its weights to view the results.

## Results

Only two states were used to train the model, that is the distance between the bird and the next pipe along the X and the Y axes. The Q-learning model was trained over 40000 episodes and achieved a peak score of 480 with an average of 64 over the last 2000 episodes. The DQN was trained over roughly 20000 episodes (so far) and achieved a peak score of 800 and an average of over 250.

10000 episodes: 



https://github.com/AlThorStormblessed/Q_Learning_Flappy_Bird/assets/86941904/43487691-c516-456f-a797-f454ffce9b13




40000 episodes:



https://github.com/AlThorStormblessed/Q_Learning_Flappy_Bird/assets/86941904/5fb6fe5a-3607-4575-8adf-f7bde917d2a5



The model gradually learns how to play the game and begins to achieve higher and higher totals, going as high as 500. Using states such as the bird's absolute height, speed, position of the next to next pipe as well, etc will increase training time and complexity but will also rapidly increase scores.


## Deep Q Learning model

The average score of the game was increased greatly by the use of NN, to over 250. The following is an example of a run:

https://drive.google.com/file/d/1M4LbFsHekO2BPjBapU8I_q_A5CY1UVaM/view?usp=sharing

