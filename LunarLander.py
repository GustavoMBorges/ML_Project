from DeepQLearning_Gu import DQN
import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model


import pickle
from matplotlib import pyplot as plt


def test_already_trained_model(trained_model):
    rewards_list = []
    num_test_episode = 100
    env = gym.make("LunarLander-v2")
    print("Teste do modelo treinado...")

    step_count = 1000

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        reward_for_episode = 0
        for step in range(step_count):
            # env.render()
            selected_action = np.argmax(
                trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode += reward
            if done:
                break
        rewards_list.append(reward_for_episode)
        print(test_episode, "\t: Episodios || Recompensa: ", reward_for_episode)

    return rewards_list


def plot_df(df, chart_name, title, x_axis_label, y_axis_label):
    plt.rcParams.update({'font.size': 17})
    df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    # plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    # plt.ylim((-400, 300))
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)


def plot_df2(df, chart_name, title, x_axis_label, y_axis_label):
    df['mean'] = df[df.columns[0]].mean()
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    # plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim((0, 300))
    plt.xlim((0, 100))
    plt.legend().set_visible(False)
    fig = plot.get_figure()
    fig.savefig(chart_name)


def plot_experiments(df, chart_name, title, x_axis_label, y_axis_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1, figsize=(15, 8), title=title)
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim(y_limit)
    fig = plot.get_figure()
    fig.savefig(chart_name)


def run_experiment_for_gamma():
    print('Otimização de gamma...')
    env = gym.make('LunarLander-v2')

    # set seeds
    env.seed(21)
    np.random.seed(21)

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma_list = [0.99, 0.9, 0.8, 0.7]
    training_episodes = 1000

    rewards_list_for_gammas = []
    for gamma_value in gamma_list:
        # save_dir = "hp_gamma_"+ str(gamma_value) + "_"
        model = DQN(env, lr, gamma_value, epsilon, epsilon_decay)
        print("Treinando o modelo para gamma igual a: {}".format(gamma_value))
        model.train(training_episodes, False)
        rewards_list_for_gammas.append(model.rewards_list)

    pickle.dump(rewards_list_for_gammas, open(
        "rewards_list_for_gammas.p", "wb"))
    rewards_list_for_gammas = pickle.load(
        open("rewards_list_for_gammas.p", "rb"))

    gamma_rewards_pd = pd.DataFrame(
        index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(gamma_list)):
        col_name = "gamma=" + str(gamma_list[i])
        gamma_rewards_pd[col_name] = rewards_list_for_gammas[i]
    plot_experiments(gamma_rewards_pd, "Recompensa por Episodio (Variando Gamma)",
                     "Recompensa por Episodio (Variando Gamma)", "Episodios", "Recompensas", (-600, 300))


def run_experiment_for_lr():
    print('Otimização de Alpha...')
    env = gym.make('LunarLander-v2')

    # set seeds
    env.seed(21)
    np.random.seed(21)

    # setting up params
    lr_values = [0.0001, 0.001, 0.01, 0.1]
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 1000
    rewards_list_for_lrs = []
    for lr_value in lr_values:
        model = DQN(env, lr_value, gamma, epsilon, epsilon_decay)
        print("Treinando o modelo para alpha igual a: {}".format(lr_value))
        model.train(training_episodes, False)
        rewards_list_for_lrs.append(model.rewards_list)

    pickle.dump(rewards_list_for_lrs, open("rewards_list_for_lrs.p", "wb"))
    rewards_list_for_lrs = pickle.load(open("rewards_list_for_lrs.p", "rb"))

    lr_rewards_pd = pd.DataFrame(
        index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(lr_values)):
        col_name = "lr=" + str(lr_values[i])
        lr_rewards_pd[col_name] = rewards_list_for_lrs[i]
    plot_experiments(lr_rewards_pd, "Recompensa por Episodio (Variando Alpha)",
                     "Recompensa por Episodio (Variando Alpha)", "Episodios", "Recompensas", (-2000, 300))


def run_experiment_for_ed():
    print('Otimização do decaimento do Episilon...')
    env = gym.make('LunarLander-v2')

    # set seeds
    env.seed(21)
    np.random.seed(21)

    # setting up params
    lr = 0.001
    epsilon = 1.0
    ed_values = [0.999, 0.995, 0.990, 0.9]
    gamma = 0.99
    training_episodes = 1000

    rewards_list_for_ed = []
    for ed in ed_values:
        save_dir = "hp_ed_" + str(ed) + "_"
        model = DQN(env, lr, gamma, epsilon, ed)
        print("Treinando o modelo para decaimento d episilon igual a: {}".format(ed))
        model.train(training_episodes, False)
        rewards_list_for_ed.append(model.rewards_list)

    pickle.dump(rewards_list_for_ed, open("rewards_list_for_ed.p", "wb"))
    rewards_list_for_ed = pickle.load(open("rewards_list_for_ed.p", "rb"))

    ed_rewards_pd = pd.DataFrame(
        index=pd.Series(range(1, training_episodes+1)))
    for i in range(len(ed_values)):
        col_name = "epsilon_decay = " + str(ed_values[i])
        ed_rewards_pd[col_name] = rewards_list_for_ed[i]
    plot_experiments(ed_rewards_pd, "Recompensa por Episodio (Variando Decaimento de epsilon)",
                     "Recompensa por Episodio (Variando Decaimento de epsilon)", "Episodios", "Recompensa", (-600, 300))


env = gym.make('LunarLander-v2')

# define semente
# env.seed(21)
np.random.seed(21)

# define parametros
lr = 0.001
epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.99
training_episodes = 2000
print('St')
model = DQN(env, lr, gamma, epsilon, epsilon_decay)
model.train(training_episodes, True)

# Salva
save_dir = "data/"
# Modelo
model.save(save_dir + "trained_model.h5")

# Salva lista de recomepnça
pickle.dump(model.rewards_list, open(save_dir + "Recompensas_treino.p", "wb"))
rewards_list = pickle.load(open(save_dir + "Recompensas_treino.p", "rb"))

# Plotar Recompensa x por Epiódio de Treino
reward_df = pd.DataFrame(rewards_list)
plot_df(reward_df, "Recompensa x por Epiódio de Treino",
        "Recompensa por episodios de Treino", "Episodeos", "Recompensa")

# Testa o modelo
trained_model = load_model(save_dir + "trained_model.h5")
test_rewards = test_already_trained_model(trained_model)
pickle.dump(test_rewards, open(save_dir + "Recompensas_teste.p", "wb"))
test_rewards = pickle.load(open(save_dir + "Recompensas_teste.p", "rb"))

# Plotar Recompensa x por Epiódio de Teste
plot_df2(pd.DataFrame(test_rewards), "Recompensa x por Epiódio de Teste",
         "Recompensa por episodios de Teste", "Episodeos", "Recompensa")
print("Fim do Treino e Teste !")

# Run experiments for hyper-parameter
run_experiment_for_lr()
run_experiment_for_ed()
run_experiment_for_gamma()
