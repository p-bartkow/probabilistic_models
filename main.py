from itertools import product
import numpy as np
import matplotlib.pyplot as plt

F = 0
T = 1

p_burglary = np.zeros(2)
p_burglary[T] = 0.001
p_burglary[F] = 1 - p_burglary[T]
p_thunderbolt = np.zeros(2)
p_thunderbolt[T] = 0.002
p_thunderbolt[F] = 1 - p_thunderbolt[T]

p_alarm_b_t = np.zeros((2, 2, 2))
p_alarm_b_t[T, T, T] = 0.95
p_alarm_b_t[T, F, T] = 0.94
p_alarm_b_t[F, T, T] = 0.29
p_alarm_b_t[F, F, T] = 0.001
p_alarm_b_t[T, T, F] = 0.05
p_alarm_b_t[T, F, F] = 0.06
p_alarm_b_t[F, T, F] = 0.71
p_alarm_b_t[F, F, F] = 0.999

p_stefan_a = np.zeros((2, 2))
p_stefan_a[T, T] = 0.9
p_stefan_a[T, F] = 0.1
p_stefan_a[F, T] = 0.05
p_stefan_a[F, F] = 0.95

p_barbara_a = np.zeros((2, 2))
p_barbara_a[T, T] = 0.7
p_barbara_a[T, F] = 0.3
p_barbara_a[F, T] = 0.01
p_barbara_a[F, F] = 0.99

# Generating a table of the total probability distribution
t = np.zeros((2, 2, 2, 2, 2))
for W in [T, F]:
    for X in [T, F]:
        for A in [T, F]:
            for S in [T, F]:
                for B in [T, F]:
                    t[W, X, A, S, B] = \
                        p_burglary[W] * p_thunderbolt[X] * p_alarm_b_t[W, X, A] * p_stefan_a[A, S] * p_barbara_a[A, B]

# Alarm if both neighbors rang
p_s_and_b = sum([t[w, x, a, T, T] for w, x, a in product([T, F], [T, F], [T, F])])
p_a_sb = sum([t[w, x, T, T, T] for w, x in product([T, F], [T, F])])
p_a_if_sb_exact = p_a_sb / p_s_and_b
print('Alarm if both neighbors rang exact: ', p_a_if_sb_exact)

# Burglary if both neighbors rang
p_s_and_b = sum([t[w, x, a, T, T] for w, x, a in product([T, F], [T, F], [T, F])])
p_b_sb = sum([t[T, x, a, T, T] for x, a in product([T, F], [T, F])])
p_b_if_sb_exact = p_b_sb / p_s_and_b
print('Burglary if both neighbors rang exact: ', p_b_if_sb_exact)

# Calculation of selected conditional probabilities using the Monte Carlo method

iterations = 100  # Number of iterations
samples = 2000  # Number of samples for irrigation

avg_p_a_if_sb = 0
avg_p_b_if_sb = 0

plt.axis([0, iterations, 0, 2])
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.plot([0, iterations], [p_a_if_sb_exact, p_a_if_sb_exact], color='b')
plt.plot([0, iterations], [p_b_if_sb_exact, p_b_if_sb_exact], color='b')

np.random.seed(1)
for i in range(1, iterations):
    # Generating the values of individual random variables for the random course of the network
    b = np.random.random(samples) < p_burglary[T]
    t = np.random.random(samples) < p_thunderbolt[T]
    aTT = np.random.random(samples) < p_alarm_b_t[T, T, T]
    aTF = np.random.random(samples) < p_alarm_b_t[T, F, T]
    aFT = np.random.random(samples) < p_alarm_b_t[F, T, T]
    aFF = np.random.random(samples) < p_alarm_b_t[F, F, T]
    a = np.logical_or.reduce((
        np.logical_and.reduce((b, t, aTT)),
        np.logical_and.reduce((b, np.logical_not(t), aTF)),
        np.logical_and.reduce((np.logical_not(b), t, aFT)),
        np.logical_and.reduce((np.logical_not(b), np.logical_not(t), aFF))
    ))

    stefanT = np.random.random(samples) < p_stefan_a[T, T]
    stefanF = np.random.random(samples) < p_stefan_a[F, T]
    stefan = np.logical_or(
        np.logical_and(a, stefanT),
        np.logical_and(np.logical_not(a), stefanF)
    )

    barbaraT = np.random.random(samples) < p_barbara_a[T, T]
    barbaraF = np.random.random(samples) < p_barbara_a[F, T]
    barbara = np.logical_or(
        np.logical_and(a, barbaraT),
        np.logical_and(np.logical_not(a), barbaraF)
    )

    # P(A = P | S = T, B = T) = P(A = T, S = T, B = T) / P(S = T, B = T)
    p_a_if_sb__mc = np.sum(np.logical_and.reduce((a, stefan, barbara))) / np.sum(
        np.logical_and.reduce((stefan, barbara)))

    # P(W = P | S = T, B = T) = P(W = T, S = T, B = T) / P(S = T, B = T)
    p_w_if_sb__mc = np.sum(np.logical_and.reduce((b, stefan, barbara))) / np.sum(
        np.logical_and.reduce((stefan, barbara)))

    # From the iterative formula for calculating the mean
    avg_p_a_if_sb = avg_p_a_if_sb + (p_a_if_sb__mc - avg_p_a_if_sb) / i
    avg_p_b_if_sb = avg_p_b_if_sb + (p_w_if_sb__mc - avg_p_b_if_sb) / i

    plt.scatter(i, avg_p_a_if_sb, marker='.', s=1, color='r')
    plt.scatter(i, avg_p_b_if_sb, marker='.', s=1, color='r')

    plt.pause(0.001)

print('Alarm if both neighbors rang Monte Carlo: ', avg_p_a_if_sb)
print('Burglary if both neighbors rang Monte Carlo: ', avg_p_b_if_sb)
