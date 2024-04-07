import matplotlib.pyplot as plt
import pygame, random, time, math, numpy as np, pickle, os
from pygame.locals import *
from matplotlib import style
import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten
)
from collections import deque
from tqdm import tqdm

style.use("ggplot")


#VARIABLES
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
n = 3
SPEED = 20
GRAVITY = 10
GAME_SPEED = 15


GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT= 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500

PIPE_GAP = 150

class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images =  [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]

        self.speed = SPEED

        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        # self.speed += GRAVITY
        self.speed = min(min(0, self.speed) + GRAVITY, SPEED)

        #UPDATE HEIGHT
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -2 * SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))


        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize


        self.mask = pygame.mask.from_surface(self.image)


    def update(self):
        self.rect[0] -= GAME_SPEED

        

class Ground(pygame.sprite.Sprite):
    
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT
    def update(self):
        self.rect[0] -= GAME_SPEED

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(25, 45) * 10
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    flag_ = True
    return pipe, pipe_inverted

def disc_pos(pos):
    pos = int(pos//5) * 5
    return pos
    
def imp(pos):
    pos = int(pos//10) * 10
    return pos

def Capture(display,name,pos,size): # (pygame Surface, String, tuple, tuple)
    image = pygame.Surface(size)  # Create image surface
    image.blit(display,(0,0),(pos,size))  # Blit portion of the display to the image
    pygame.image.save(image,name) 


rewards = []
num_ep = 100000
score_rew = 15
train_every = 15
flag_ = True
crash = -1000
over_flow = -1000
epsilon = 0.4
epsilon_decay = .9998
alpha = 0.05
# alpha_decay = 0.9998
discount = 0.99
show_every = 10
peak_score = 0
mb_size = 100
D = deque([], maxlen=10000)   

model = Sequential()
model.add(Dense(16, input_shape=(4,80,80), activation='relu'))
model.add(Flatten())       # Flatten input so as to have no problems with processing
# model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))    # Same number of outputs as possible actions
checkpoint_path = "training_2/cp_100000_episodes.ckpt"
model.load_weights(checkpoint_path)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

scores = []

for i in range(num_ep):
    if i % show_every == 0:
        show = True 
    else:
        show = False

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Flappy Bird')

    BACKGROUND = pygame.image.load('assets/sprites/blank.png')
    BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))

    bird_group = pygame.sprite.Group()
    bird = Bird()
    bird_group.add(bird)

    ground_group = pygame.sprite.Group()

    for j in range (2):
        ground = Ground(GROUND_WIDTH * j)
        ground_group.add(ground)

    pipe_group = pygame.sprite.Group()
    for j in range(2):
        pipes = get_random_pipes(SCREEN_WIDTH * j + 300)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    clock = pygame.time.Clock()
    score = 0

    clock.tick(500)
    bird.bump()

    screen.blit(BACKGROUND, (0, 0))

    if is_off_screen(ground_group.sprites()[0]): 
        ground_group.remove(ground_group.sprites()[0])

        new_ground = Ground(GROUND_WIDTH - 20)
        ground_group.add(new_ground)

    bird.begin()
    ground_group.update()

    bird_group.draw(screen)
    ground_group.draw(screen)

    pygame.display.update()

    image = pygame.Surface((300, 500))
    image = pygame.transform.scale(image, (80, 80))
    obs = pygame.surfarray.array3d(image)
    obs = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])
    obs = np.expand_dims(obs, axis=0)

    state = np.stack((obs, obs, obs, obs), axis = 1)

    total_reward = 0
    while True:

        clock.tick(500)

        if random.random() > epsilon: 
            action = random.randint(0, 1)
        else:
            Q = model.predict(state, verbose = 0)          # Q-values predictions
            action = np.argmax(Q)   
            # if action == 0: print(action)

        if(action): 
            bird.bump()

        screen.blit(BACKGROUND, (0, 0))

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])

            new_ground = Ground(GROUND_WIDTH - 20)
            ground_group.add(new_ground)

        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])

            pipes = get_random_pipes(SCREEN_WIDTH * 2)

            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

        bird_group.update()
        ground_group.update()
        pipe_group.update()

        bird_group.draw(screen)
        pipe_group.draw(screen)
        ground_group.draw(screen)

        pygame.display.update()

        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
            reward = crash
        elif(bird.rect[1] < 0):
            reward = over_flow

        else:
            # print(pipe_group.sprites()[0].rect)
            if(pipe_group.sprites()[0].rect[0] < bird.rect[0] and flag_):
                score += 1
                flag_ = False
                peak_score = max(peak_score, score)
                reward = score_rew
            elif(pipe_group.sprites()[0].rect[0] > bird.rect[0]):
                flag_ = True
                reward = 15
            else:
                reward = 15
        
        # alpha = 1/(1 + q_table[obs][2])
        image = pygame.Surface((300, 500))
        image = pygame.transform.scale(image, (80, 80))
        new_obs = pygame.surfarray.array3d(image)
        new_obs = np.dot(new_obs[...,:3], [0.2989, 0.5870, 0.1140])
        new_obs = np.expand_dims(new_obs, axis=0)
        state_new = np.append(np.expand_dims(new_obs, axis=0), state[:, :3, :], axis=1)
        
        # print(state_new.shape)
        
        
        # max_future_q = np.max(q_table[new_obs][:2])
        # current_q = q_table[obs][action]
        
        # new_q = (1 - alpha) * current_q + alpha * (reward + discount * max_future_q)
        # q_table[obs][action] = new_q

        if reward == crash or reward == over_flow:
            done = True
        else:
            done = False

        D.append((state, action, reward, state_new, done))

        q = model.predict(state, verbose = 0)
        qn = model.predict(state_new, verbose = 0)

        if done:
            target = reward
        else:
            target = reward + discount * np.max(qn[0])

        # alpha = 1/(1 + D.count((state, action, reward, state_new, done)))
        q[0][action] = target
        model.fit(state, q, epochs=1, verbose=0)
    
        state = state_new

        total_reward += reward

        if(reward == crash or reward == over_flow):
            # if show:
            #     pygame.mixer.music.load(hit)
            #     pygame.mixer.music.play()
            #     time.sleep(2)
            break
    
    if show:
        try: 
            minibatch = random.sample(D, mb_size)
            inputs_shape = (mb_size,) + state.shape[1:]
            inputs = np.zeros(inputs_shape)
            targets = np.zeros((mb_size, 2))

            for j in range(0, mb_size):
                state_D = minibatch[j][0]
                action_D = minibatch[j][1]
                reward_D = minibatch[j][2]
                state_new_D = minibatch[j][3]
                done_D = minibatch[j][4]
            # Build Bellman equation for the Q function
                inputs[j:j+1] = np.expand_dims(state_D, axis=0)
                targets[j] = model.predict(state_D, verbose = 0)
                Q_sa = model.predict(state_new_D, verbose = 0)
                
                if done_D:
                    targets[j, action_D] = reward_D
                else:
                    targets[j, action_D] = reward_D + discount * np.max(Q_sa)

            # Train network to output the Q function
            model.train_on_batch(inputs, targets)

        except: 
            pass

    scores.append(score)
    epsilon *= epsilon_decay
    if show: 
        print(f"{i + 1}th Episode: Reward = {total_reward}, Peak Score = {peak_score}, Score = {score}, Rolling Average = {round(np.mean(scores[-200:]), 4)}")
        rewards.append(round(np.mean(scores[-show_every:]), 4))

        model.save_weights("training_2/cp_100000_episodes.ckpt")

print(rewards)
