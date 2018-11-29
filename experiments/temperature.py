import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])
T = np.linspace(0.01, 5, num=100)
P = np.empty([len(X), len(T)])
alpha = min(X)

for t, ind in zip(T, range(len(T))):
    sum = 0
    for i in range(len(X)):
        numerator = (X[i] / alpha) ** (-1 / t)
        P[i, ind] = numerator

        sum += numerator

    P[:, ind] /= sum

#print(P[:, 10]) # sanity check for the closest to example t value - 0.51...

for i in range(len(X)):
    plt.plot(T, P[i, :], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
