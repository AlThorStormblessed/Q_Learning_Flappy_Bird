import matplotlib.pyplot as plt
import pygame, random, time, math, numpy as np, pickle
from pygame.locals import *
from matplotlib import style
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten
)
import os

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

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'

pygame.mixer.init()


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

num_ep = 1000
score_rew = 15
flag_ = True
crash = -1000
over_flow = -1000
epsilon = 0.
epsilon_decay = .98
alpha = 0.05
alpha_decay = 0.9998
discount = 1
show_every = 20000
peak_score = 0
scores = []

model = Sequential()
model.add(Dense(32, input_shape=(2,) + np.array((0, 0)).shape, activation='relu'))
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='linear'))    # Same number of outputs as possible actions
checkpoint_path = "training_2/cp2.weights.h5"
model.load_weights(checkpoint_path)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

for i in range(num_ep):
    os.mkdir(f"{i + 1}")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Flappy Bird')

    BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
    BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
    digits = [pygame.image.load(f'assets/sprites/{i}.png') for i in range(10)]

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
    begin = True

    while begin:

        clock.tick(240)

        bird.bump()
        pygame.mixer.music.load(wing)
        pygame.mixer.music.play()
        begin = False

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

    obs = (imp(pipe_group.sprites()[1].rect[0]), imp(pipe_group.sprites()[0].rect[1] - bird.rect[1]))
    obs = np.expand_dims(obs, axis=0)
    state = np.stack((obs, obs), axis = 1)

    total_reward = 0
    for k in range(100000):
        obs = (imp(pipe_group.sprites()[1].rect[0]), imp(pipe_group.sprites()[0].rect[1] - bird.rect[1]))

        clock.tick(240)

        action = np.argmax(model.predict(state, verbose = 0)   )
            # print("Action")

        if(action): 
            bird.bump()
            pygame.mixer.music.load(wing)
            pygame.mixer.music.play()

        screen.blit(BACKGROUND, (0, 0))
        x = 130
        for num in str(score):
            screen.blit(digits[int(num)], (x, 120))
            x += 20

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

        Capture(screen, f"{i + 1}/{k}.png", (0, 0), (300, 600))

        # print(bird.speed)

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
        
        new_obs = (imp(pipe_group.sprites()[1].rect[0]), pipe_group.sprites()[0].rect[1] - disc_pos(bird.rect[1]))
        new_obs = np.expand_dims(new_obs, axis=0)
        state = np.append(np.expand_dims(new_obs, axis=0), state[:, :1, :], axis=1)

        if(reward == crash or reward == over_flow):
            pygame.mixer.music.load(hit)
            pygame.mixer.music.play()
            time.sleep(2)
            break
    
    scores.append(score)
    print(f"{i + 1}th Episode: Reward = {total_reward}, Peak Score = {peak_score}, Score = {score}, Rolling Average = {round(np.mean(scores[-200:]), 4)}")
    
print("Hello world")
