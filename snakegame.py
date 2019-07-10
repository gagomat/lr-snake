from random import randint
from tkinter import *

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np


class Snakegame():

    def __init__(self, w=10, h=10):
        self.width = w
        self.height = h
        self.snake = list()
        self.seed = None
        self.env = None
        self.r = 0.
        self.end = 0
        self.action = ''
        self.istep = 0

        #initialisation du jeu
        self.reset()

    def reset(self):
        """reset du jeu"""
        self.istep = 0
        self.snake = list()
        x, y = self._random_pos()
        self.snake.append((x, y))
        self._new_feed()
        self._end_step()

    def step(self, action):
        s = self.snake
        h = s[0]
        self.action = action

        # calcul de la position de la nouvelle tête
        if action == 0:  # UP
            nhead = (h[0] - 1, h[1])
        elif action == 1:  # DOWN
            nhead = (h[0] + 1, h[1])
        elif action == 2:  # LEFT
            nhead = (h[0], h[1] - 1)
        elif action == 3:  # RIGHT
            nhead = (h[0], h[1] + 1)

        #check si le snake tjs dans la grille
        if ( self.env[nhead] != -1 ):
            if (nhead in s):  #check si le snake ne se mord pas
                #print('bite itself')
                self._end_step(end=1, r=-10.)
            else:
                self.snake.insert(0, nhead)
                if (nhead == self.seed):  # si même emplacement que la graine
                    self._new_feed()
                    self._end_step(end=0, r=10)

                else:
                    self.snake.pop()
                    self._end_step(end=0, r=-0.02)

        else:
            #print('out of grid')
            self._end_step(end=1, r=-10.)

        #print('action : {}, recomp : {}, end : {} '.format(action,self.r,self.end))
        #print(self.env)

    @staticmethod
    def display(buffer):

        images = buffer.images[0: buffer.i_save]
        fig = plt.figure()
        im = plt.imshow(images[0], animated=True)

        def update(istep):
            im.set_data(images[istep])
            return im

        ani = animation.FuncAnimation(fig, update)

        plt.show()

    def render(self, buffer):

        images = buffer.images[0:buffer.i_save]
        main = Tk()
        canvas = Canvas(main, width=self.width*30, height=self.height*30, background="white")
        canvas.pack(fill="both", expand=True)

        def step(i_step):
            canvas.delete('all')
            for i, l in enumerate(images[i_step]):
                for j, v in enumerate(l):
                    if v == 0.9:
                        color = 'grey'
                    elif v == 1.:
                        color = 'black'
                    elif v == 0.5:
                        color = 'green'
                    elif v == -1.:
                        color = 'blue'
                    else:
                        color = 'white'
                    rect = canvas.create_rectangle(j*20, i*20, (j+1)*20, (i+1)*20, fill=color)


            i_step += 1
            canvas.after(500, lambda: step(i_step))

        istep = 0
        btn_run = Button(main, text='run', width=20, command=lambda : step(istep))
        btn_run.pack(pady = 10)
        btn_quit = Button(main, text='quit', width=20, command=main.quit)
        btn_quit.pack()
        main.mainloop()

    def _new_feed(self):
        while True:
            x, y = self._random_pos()
            if (x, y) not in self.snake:
                self.seed = (x, y)
                break

    def _env_actu(self):
        self.old_env = self.env
        self.env = np.zeros((self.width, self.height))

        #bord de la grille a -1
        self.env[0]  = -1.
        self.env[-1] = -1.
        self.env[:,0]   = -1
        self.env[:,-1]  = -1

        for i, s in enumerate(self.snake):
            self.env[s[0]][s[1]] = 1. if i > 0 else 0.9

        self.env[self.seed[0]][self.seed[1]] = 0.5

    def _end_step(self, end=0, r=0.):
        self._env_actu()
        self.end = end
        self.r = r

    def _random_pos(self):
        return randint(1, self.width - 2), randint(1, self.height - 2)