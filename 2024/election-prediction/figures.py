import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import comb

cycler = ["#44bb55", "#2255ff", "#2255dd", "#2255bb", "#225599", "#225577", "#cc4422"][::-1]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=cycler)
mpl.rcParams.update({'font.size': 18})

N = 60
W = 5
X = np.arange(1, N+1)
y = []
for i in range(W+1):
    y_i = np.power(0.5, X)
    for j in range(N):
        y_i[j] *= comb(X[j], i)
    y.append(y_i)
y = np.array(y)
remaining = 1.0 - np.sum(y, axis=0)

plt.figure(figsize=(12, 6))
plt.title("Probability of predicting elections by guessing randomly")
plt.plot(X, remaining, label=f'>{W} wrong')
for i in reversed(range(1, W+1)):
    plt.plot(X, y[i], label=f'{i} wrong')
plt.plot(X, y[0], label='All correct')
plt.xlabel('Number of elections')
plt.ylabel('Probability')
plt.legend()
plt.savefig('figures/random.png')

plt.figure(figsize=(12,6))
plt.title("Probability of at least one person out of 300M predicting elections")
POP = 300000000
plt.plot(X, 1-np.power(1-remaining,POP), label=f'>{W} wrong')  
for i in reversed(range(1, W+1)):
    plt.plot(X, 1-np.power(1-y[i], POP), label=f'{i} wrong')
plt.plot(X, 1-np.power(1-y[0],POP), label='All correct') 
plt.xlabel('Number of elections')
plt.ylabel('Probability')
plt.legend()
plt.savefig('figures/300m.png')

plt.show()