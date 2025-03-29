import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01, 1, 100)
loss = -np.log(x)

plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(12, 6))
plt.plot(x, loss, label='Loss Function', color='blue')
plt.title('Log loss')
plt.xlabel('$p_y$ (probability of correct token)')
plt.ylabel('loss')
plt.grid()
plt.savefig('figures/log-loss.png', dpi=300, bbox_inches='tight')

x = np.linspace(-8, 8, 100)
sigmoid = 1 / (1 + np.exp(-x))
loss = -np.log(sigmoid)

plt.figure(figsize=(12, 6))
plt.plot(x, loss, label='Loss Function', color='blue')
plt.title('Log sigmoid loss')
plt.xlabel('$r_{\\theta}(x,y_w) - r_{\\theta}(x,y_l)$\n(difference in reward between winning and losing outputs)')
plt.ylabel('loss')
plt.grid()
plt.savefig('figures/rm-loss.png', dpi=300, bbox_inches='tight')