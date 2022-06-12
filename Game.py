import pygame
import numpy as np
from pygame.locals import *
import sys
 
import gym
import gym_qcart 

pygame.init()
 
vec = pygame.math.Vector2 
HEIGHT = 450
WIDTH = 400
ACC = 0.5
FRIC = -0.12
FPS = 60
RESOULTION = 1001
FramePerSec = pygame.time.Clock()
font_small = pygame.font.SysFont("Verdana", 20)
x_range = np.linspace(-10,10,RESOULTION)
function = x_range**2

displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game")
arrow_r = pygame.image.load("game_assets/arrow_r.png")
arrow_l = pygame.image.load("game_assets/arrow_l.png")
arrow_u = pygame.image.load("game_assets/arrow_u.png")
arrow_d = pygame.image.load("game_assets/arrow_d.png")


def draw_curve(function):
    curve = []
    for i in range(RESOULTION):
        curve.append( (int(i*WIDTH/RESOULTION) ,-1*function[i]*HEIGHT + HEIGHT -1) )
    return curve

class Wave2(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.action = 0
        self.x = np.linspace(-10,10,RESOULTION)
        self.env = gym.make('qgame-v0', sigma = 0.3, dt = 0.005, L = 0.02, mu = 2)
        self.env.reset()
        self.curve = draw_curve(self.env.probability_distribution)
        self.arrow = arrow_u
        self.highscore = 0
        self.score = 0
        self.done = False


    def update(self):
        pressed_keys = pygame.key.get_pressed()
        self.score += 1
        
        if pressed_keys[K_LEFT]:
            self.action = 1
            _ , _, self.done, _ = self.env.step(self.action)
            self.curve = draw_curve(self.env.probability_distribution)
            self.arrow = arrow_l
        elif pressed_keys[K_RIGHT]:
            self.action = 2
            _ , _, self.done, _ = self.env.step(self.action)
            self.curve = draw_curve(self.env.probability_distribution)
            self.arrow = arrow_r
        elif pressed_keys[K_DOWN]:
            self.action = 3
            _ , _, self.done, _ = self.env.step(self.action)
            self.curve = draw_curve(self.env.probability_distribution)
            self.arrow = arrow_d
        elif pressed_keys[K_UP]:
            self.action = 4
            _ , _, self.done, _ = self.env.step(self.action)
            self.curve = draw_curve(self.env.probability_distribution)
            self.arrow = arrow_u
            done = True                  
        else:
            self.action = 0
            _ , _, self.done, _ = self.env.step(self.action)
            self.curve = draw_curve(self.env.probability_distribution)

        if self.done == True:
            if self.score > self.highscore:
                self.highscore = self.score
            self.score = 0
            self.done = False
            self.env.reset()




W2 = Wave2()
all_sprites = pygame.sprite.Group()

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    displaysurface.fill((255, 255, 255))
    displaysurface.blit(W2.arrow, (100, 100))
    curve = draw_curve(function)
    W2.update()
    pygame.draw.lines(displaysurface, (0,0,0), False, W2.curve)
    scores = font_small.render(str(W2.score), True, (0,0,0))
    highscores = font_small.render(str(W2.highscore), True, (0,0,0))
    displaysurface.blit(scores, (10,30))
    displaysurface.blit(highscores, (10,10))

    pygame.display.update()
    FramePerSec.tick(FPS)