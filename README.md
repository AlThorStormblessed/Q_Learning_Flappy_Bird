# Q_Learning_Flappy_Bird

## Description

Following is the code for the implementation of the following paper in which a simple Q-Learning algorithm was trained using positional values to play the Flappy Bird game: https://kilyos.ee.bilkent.edu.tr/~eee546/FlappyQ.pdf

The implementation follows two variations of the game, one with constant speed while falling and the other with constant gravity.

## Code

The flappy.py and flappy_gravity.py files are used to train the Q-learning algorithm over a set number of episodes, while test.py is used to load a saved Q_table to view the results in the form of the game.

## Results

Only two states were used to train the model, that is the distance between the bird and the next pipe along the X and the Y axes. The model was trained over 40000 episodes and achieved a peak score of 480 with an average of 64 over the last 2000 episodes.

10000 episodes: 



https://github.com/AlThorStormblessed/Q_Learning_Flappy_Bird/assets/86941904/43487691-c516-456f-a797-f454ffce9b13




20000 episodes:





https://github.com/AlThorStormblessed/Q_Learning_Flappy_Bird/assets/86941904/cfb51759-3c61-4cdb-acfa-fa4a61ba12c0







30000 episodes:





https://github.com/AlThorStormblessed/Q_Learning_Flappy_Bird/assets/86941904/cfe303fc-595f-483e-8879-3695bee7a28a





40000 episodes:



https://github.com/AlThorStormblessed/Q_Learning_Flappy_Bird/assets/86941904/5fb6fe5a-3607-4575-8adf-f7bde917d2a5



The model gradually learns how to play the game and begins to achieve higher and higher totals, going as high as 500. Using states such as the bird's absolute height, speed, position of the next to next pipe as well, etc will increase training time and complexity but will also rapidly increase scores. This can also be improved by the use of Deep Q Learning.
