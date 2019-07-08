# init du jeu
import tensorflow as tf
from snakegame import Snakegame
from policynetworkagent import PolicyNetworkAgent
import numpy as np


xsize = 10
ysize = 10
buffer_size = 10000
learning_rate = 0.001
nb_recomp = 1000
nb_actions = 4

game = Snakegame(xsize, ysize)
agent = PolicyNetworkAgent(input_shape=(xsize, ysize),
                           output_shape=nb_actions,
                           lr=learning_rate,
                           buffer_size=buffer_size,
                           seed=42)
agent.compile()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    agent.set_sess(sess)

    recompenses = list()
    for epoch in range(100000):
        game.reset()
        while not game.end:

            action = agent.predict_action([game.env])
            game.step(action)
            agent.buffer.save(game.old_env, action, game.r)

            recompenses.append(game.r)
            if len(recompenses) > nb_recomp: recompenses.pop(0)

            if agent.buffer.full():
                agent.buffer.end_game()
                agent.train()
            elif game.end:
                agent.buffer.end_game()
            elif game.istep > 1000:
                game.end = 1
                agent.buffer.end_game()

        if epoch % 1000 == 0:
            print("{} : Moyenne des recompenses: {}".format(epoch, np.mean(recompenses)))

    _, _, _ = agent.buffer.get()

    for partie in range(10):
        game.reset()
        while not game.end:

            action = agent.predict_action([game.env])
            game.step(action)
            agent.buffer.save(game.old_env, action, game.r)

            if game.end:
                agent.buffer.end_game()

    game.render(agent.buffer)