import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
# Use Agg rendering engine instead of X11 for some remote servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation


model_filepath = 'dqn.hdf5'


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def build_model(env):
    s = keras.layers.Input(shape=env.observation_space.shape)
    a = keras.layers.Input(shape=1, dtype=tf.int32)
    x = keras.layers.Dense(8, activation="relu", input_shape=env.observation_space.shape)(s)
    x = keras.layers.Dense(16, activation="relu")(x)
    q_sa = keras.layers.Dense(env.action_space.n, activation="linear")(x)
    Q_net = keras.Model(inputs=[s], outputs=[q_sa], name="Q_sa_net")
    Q_net.compile('Adam', loss='mse')
    if os.path.exists(model_filepath):
        Q_net.load_weights(model_filepath)
    return Q_net


def episode(env, Q_net, epsilon, steps=100):
    s = np.zeros([steps] + list(env.observation_space.shape))
    a = np.zeros([steps, 1], dtype=np.int32)
    r = np.zeros([steps])
    s_1 = np.zeros([steps] + list(env.observation_space.shape))
    observation = env.reset()
    # frames = []
    for t in range(steps):
        # env.render()
        # frames.append(env.render(mode="rgb_array"))
        if np.random.rand() > epsilon:
            action = Q_net.predict(np.expand_dims(observation, 0)).argmax(axis=1)[0]
        else:
            action = env.action_space.sample()
        s[t] = observation
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
            reward *= -1
        a[t] = action
        r[t] = reward
        s_1[t] = observation
    # save_frames_as_gif(frames)
    env.close()
    return s, a, r, s_1


def experience_replay(Q_net, s, a, r, s_1):
    batch_size = 32
    gamma = 0.9
    epochs = 50

    q_sa_1 = Q_net.predict(s_1)
    y = np.expand_dims(r + gamma * q_sa_1.max(axis=1), 1)
    q_sa = Q_net.predict(s)
    np.put_along_axis(q_sa, a, y, axis=1)

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=model_filepath,
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=False),
        keras.callbacks.EarlyStopping(patience=5, verbose=0, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, verbose=0),
    ]
    Q_net.fit(x=s, y=q_sa,
              validation_split=0.2,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              verbose=0)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    Q_net = build_model(env)
    n_episodes = 100
    epsilon = 0.9
    for e in range(n_episodes):
        s, a, r, s_1 = episode(env, Q_net, 0, steps=500)
        q_sa_max = Q_net.predict(s).max(axis=1)
        print("Start of episode %i, average reward: %f, q_sa_max: %f" % (e, r.mean(), q_sa_max.mean()))
        s, a, r, s_1 = episode(env, Q_net, epsilon, steps=5000)
        experience_replay(Q_net, s, a, r, s_1)
        epsilon = max(0.1, epsilon - 0.01)
