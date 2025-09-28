# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data = pd.read_csv(r'D:\train.csv')
data = data.dropna()

x_data1 = data.loc[:, ['as', 'de', 'rainfall', 'SPI', 'vegetation', 'distance_roads', 'NDBI', 'NDVI', 'soil', 'lithology', 'aspect', 'slope', 'elevation']]
y_data1 = data.loc[:, 'type']

y_data1 = pd.get_dummies(y_data1).values

transfer = StandardScaler()
x_data1 = transfer.fit_transform(x_data1)

x_train, x_test, y_train, y_test = train_test_split(x_data1, y_data1, random_state=1, train_size=0.7)

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

n_wolves = 10
n_iterations = 20
pa = 0.25

def generate_wolves(n_wolves):
    wolves = []
    for _ in range(n_wolves):
        hidden_neurons = random.randint(10, 100)
        learning_rate = random.uniform(0.0001, 0.01)
        wolves.append([hidden_neurons, learning_rate])
    return wolves

def fitness(hidden_neurons, learning_rate):
    hidden_neurons = int(hidden_neurons)
    model = Sequential()
    model.add(Input(shape=(1, x_train.shape[2])))
    model.add(SimpleRNN(hidden_neurons, activation='relu', return_sequences=False, stateful=False))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['AUC'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    gailv1 = model.predict(x_test)
    auc = roc_auc_score(y_test, gailv1, multi_class='ovr')
    return auc

def levy_flight(Lambda):
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                      (math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, 2)
    v = np.random.normal(0, sigma2, 2)
    step = u / np.power(np.abs(v), 1 / Lambda)
    return step

def clip_wolf(wolf):
    hidden_neurons = max(10, min(int(round(wolf[0])), 100))
    learning_rate = max(0.0001, min(float(wolf[1]), 0.01))
    return [hidden_neurons, learning_rate]

def LIGWO():
    wolves = generate_wolves(n_wolves)
    alpha, beta, delta = sorted(wolves, key=lambda w: fitness(*w), reverse=True)[:3]
    best_fitness = fitness(*alpha)
    history = []

    for t in range(n_iterations):
        a_t = 2 - 2 * (t / n_iterations) ** 3

        step_size = levy_flight(1.5)
        for i, wolf in enumerate([alpha, beta, delta]):
            if i == 0:
                wolf = [wolf[0] + step_size[0], wolf[1] + step_size[1]]
            else:
                wolf = [wolf[0] + step_size[0]*random.random(), wolf[1] + step_size[1]*random.random()]
            wolves[i] = clip_wolf(wolf)

        for i in range(n_wolves):
            for j in range(2):
                r = random.random()
                A = 2 * r * a_t - a_t
                C = 2 * random.random()

                D_alpha = abs(C * alpha[j] - wolves[i][j])
                D_beta  = abs(C * beta[j] - wolves[i][j])
                D_delta = abs(C * delta[j] - wolves[i][j])

                X1 = alpha[j] - A * D_alpha
                X2 = beta[j] - A * D_beta
                X3 = delta[j] - A * D_delta

                w1 = 1 + t / n_iterations
                w2, w3 = 1, 1
                wolves[i][j] = (w1*X1 + w2*X2 + w3*X3)/(w1 + w2 + w3)

            wolves[i] = clip_wolf(wolves[i])

            curr_fit = fitness(*wolves[i])

            avg_fit = np.mean([fitness(*w) for w in wolves])

            if curr_fit < avg_fit:
                if t < n_iterations/2:
                    wolves[i] = [clip_wolf([100 - wolves[i][0], 0.01 - wolves[i][1]])[0],
                                 clip_wolf([100 - wolves[i][0], 0.01 - wolves[i][1]])[1]]
                else:
                    wolves[i] = [(alpha[0] + wolves[i][0])/2, (alpha[1] + wolves[i][1])/2]
                wolves[i] = clip_wolf(wolves[i])

            if t % 2 == 0:
                k1, k2 = 0.5, 0.5
                wolves[i] = [k1*alpha[0] + k2*wolves[i][0], k1*alpha[1] + k2*wolves[i][1]]
            else:
                r1, r2 = random.randint(0, n_wolves-1), random.randint(0, n_wolves-1)
                Xr1, Xr2 = wolves[r1], wolves[r2]
                factor = 0.5 + 0.5*random.random()
                wolves[i] = [alpha[0] + factor*(Xr1[0]-Xr2[0]), alpha[1] + factor*(Xr1[1]-Xr2[1])]
            wolves[i] = clip_wolf(wolves[i])

            P = 0.25
            if random.random() < P:
                wolves[i][0] += np.random.randn()
                wolves[i][1] += np.random.randn()
            wolves[i] = clip_wolf(wolves[i])

        alpha, beta, delta = sorted(wolves, key=lambda w: fitness(*w), reverse=True)[:3]
        best_fitness = max(best_fitness, fitness(*alpha))
        history.append(best_fitness)
        print(f'Iteration {t+1}, Best Fitness: {best_fitness}')

    return alpha, history


best_wolf, history = LIGWO()
print('Best parameters found:', best_wolf)

best_hidden_neurons, best_learning_rate = best_wolf
model = Sequential()
model.add(Input(shape=(1, x_train.shape[2])))
model.add(SimpleRNN(int(best_hidden_neurons), activation='relu', return_sequences=False, stateful=False))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='categorical_crossentropy', metrics=['AUC'])
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

gailv1 = model.predict(x_test)
ypredict1 = np.argmax(gailv1, axis=1)
y_test_true = np.argmax(y_test, axis=1)

print('AUC', roc_auc_score(y_test, gailv1, multi_class='ovr'))
classreport = classification_report(y_test_true, ypredict1)
print(classreport)

joblib.dump(model, 'LIGWO_Elman_model.pkl')

fpr = {}
tpr = {}
plt.figure()
for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], gailv1[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (area = %0.2f)' % roc_auc_score(y_test[:, i], gailv1[:, i]))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

conf_matrix = confusion_matrix(y_test_true, ypredict1)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

plt.figure()
samples = range(len(y_test_true))
plt.scatter(samples, y_test_true, alpha=0.3, label='Actual Value')
plt.scatter(samples, np.argmax(gailv1, axis=1), alpha=0.3, label='Predicted Value', color='red')
plt.xlabel('Sample Number')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs Predicted Probability')
plt.savefig('actual_vs_predicted.png')
plt.close()
