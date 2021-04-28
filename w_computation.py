from mpmath import *
from scipy import special
import math
import scipy.integrate as integrate
import numpy as np


def double_factorial(n):
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)


# only for m = 1 and 2
def BI(n, m=1, scenario='o'):
    # n - Bessel Function type
    # senario - 'o' for original , 'p' for power series , 'm' for asymptotic
    # series
    # m - orders used for approximation

    # Combination of 1st order P.S and A.S
    if (scenario == 'o'):
        Bessel = lambda t: special.iv(n, t)
        # Bessel = lambda t: besseli(n, t)
        return Bessel

    if (scenario == 'p'):
        I_PS = lambda t, k: ((t / 2) ** n) * (1 / (gamma(k + 1) * gamma(k + n + 1))) * ((t / 2) ** (2 * k))
        if m == 1:
            B = lambda t: I_PS(t, 0) + I_PS(t, 1)
        if m == 2:
            B = lambda t: I_PS(t, 0) + I_PS(t, 1) + I_PS(t, 2)
        Bessel = lambda t: B(t)
        return Bessel

    if (scenario == 'a'):
        I_AS = lambda t, k: ((1 / (2 * pi * t)) ** (1 / 2)) * math.exp(t) * (
                    ((-1) ** k) * (n ** (2 * k)) / ((double_factorial(2 * k) * (t ** (k)))))
        if m == 1:
            B = lambda t: I_AS(t, 0) + I_AS(t, 1) + I_AS(t, 2)
        if m == 2:
            B = lambda t: I_AS(t, 0) + I_AS(t, 1) + I_AS(t, 2) + I_AS(t, 3)
        Bessel = lambda t: B(t)
        return Bessel

    if (scenario == 'c'):
        B = lambda t: besseli(n, t)
        if m == 1:
            I_AS = lambda t, k: (1 / (2 * pi * t)) ** (1 / 2) * math.exp(t) * (
                        ((-1) ** k) * (n ** (2 * k)) / ((double_factorial(2 * k) * (t ** (2 * k)))))
            B_AS = lambda t: I_AS(t, 0) + I_AS(t, 1)
            I_PS = lambda t, k: ((t / 2) ** n) * (1 / (gamma(k + 1) * gamma(k + n + 1)) * (t / 2) ** (2 * k))
            B_PS = lambda t: I_PS(t, 0) + I_PS(t, 1)
        if m == 2:
            I_AS = lambda t, k: (1 / (2 * pi * t)) ** (1 / 2) * math.exp(t) * (
                        ((-1) ** k) * (n ** (2 * k)) / ((double_factorial(2 * k) * (t ** (2 * k)))))
            B_AS = lambda t: I_AS(t, 0) + I_AS(t, 1) + I_AS(t, 2)
            I_PS = lambda t, k: ((t / 2) ** n) * (1 / (gamma(k + 1) * gamma(k + n + 1)) * (t / 2) ** (2 * k))
            B_PS = lambda t: I_PS(t, 0) + I_PS(t, 1) + I_PS(t, 2)
        Bessel = lambda t: (abs(B_AS(t) - B(t)) < abs(B_PS(t) - B(t))) * B_AS(t) + (
                    abs(B_AS(t) - B(t)) >= abs(B_PS(t) - B(t))) * B_PS(t)
        return Bessel

    Bessel = 0


# order = 1 only supported
def FQ(lam, mu, r0, r1, x, Senario='o', Order=1, t_n=20, dt=0.5):
    # constants
    a = ((lam * r1) - (mu * r0)) / (r1 - r0)
    b = (-1 / 2) * ((1 / r0) - (1 / r1))
    alpha = (-4 * lam * mu * r0 * r1 / ((r1 - r0) ** 2)) ** (1 / 2)

    # lambda
    K = lambda t: ((t + 2 * b * x) / t) ** (1 / 2)

    I_0 = BI(0, Order, Senario)
    I_1 = BI(1, Order, Senario)
    I_2 = BI(2, Order, Senario)

    Convol_left = lambda t: math.exp((lam + mu) * t)

    g = lambda t: (1 / 2) * alpha * math.exp(-1 * a * t) * (K(t) - (1 / K(t))) * I_1(alpha * K(t) * t)
    Convol_g = lambda t: Convol_left(t) * g(t)
    Integral_g_1 = lambda t: integrate.quad(g, 0, t)[0]
    Integral_g_2 = lambda t: -1 * math.exp(-1 * (lam + mu) * (t - x / r1)) * integrate.quad(Convol_g, 0, t)[0]
    Integral_g = lambda t: 1 - math.exp(-1 * (lam + mu) * (t - x / r1)) + Integral_g_1(t) + Integral_g_2(t)

    F_1_first = lambda t: (lam / (lam + mu)) * (1 - math.exp(-1 * (lam + mu) * t))
    F_1_second = lambda t: -1 * (lam / (lam + mu)) * math.exp(-1 * mu * x / r1) * Integral_g(t)

    F_1 = []
    vec = [i for i in range(0, round(t_n / dt))]
    vec = [i / 2 for i in vec]
    for t in vec:
        value = F_1_first(t)
        if t > x / r1:
            value += F_1_second(t)
        F_1.append(value)

    h = lambda t: ((lam * mu * r1) / (r1 - r0)) * math.exp(-1 * a * t) * (
                I_0(alpha * K(t) * t) - (K(t) ** (-2)) * I_2(alpha * K(t) * t))
    Convol_h = lambda t: Convol_left(t) * h(t)
    Integral_h_1 = lambda t: integrate.quad(h, 0, t)[0]
    Integral_h_2 = lambda t: -1 * math.exp(-1 * (lam + mu) * (t - x / r1)) * integrate.quad(Convol_h, 0, t)[0]

    F_0_first = lambda t: (mu / (lam + mu) + lam / (lam + mu) * math.exp(-1 * (lam + mu) * t))
    F_0_second = lambda t: -1 * (1 / (lam + mu)) * math.exp(-1 * mu * x / r1) * (Integral_h_1(t) + Integral_h_2(t))

    F_0 = []
    for t in vec:
        value = F_0_first(t)
        if t > x / r1:
            value += F_0_second(t)
        F_0.append(value)

    W = np.array(F_0) + np.array(F_1)

    return W, F_0, F_1


def MonteCarloSimulationFQ(lam, mu, r_0, r_1, x, t_end=20, dt=0.5, u=0, s0=0, iter=1000):
    r = [r_0, r_1]
    lam = [lam, mu]

    W = np.zeros(round(t_end / dt + 1))
    F1 = np.zeros(round(t_end / dt + 1))
    F2 = np.zeros(round(t_end / dt + 1))
    for k in range(0, iter):
        S = np.zeros(round(t_end / dt + 1), dtype=int)
        S[0] = s0
        Q = np.zeros(round(t_end / dt + 1))
        Q[0] = u

        # Simulation of Q(t)

        for i in range(1, round(t_end / dt + 1)):
            p = np.random.exponential(1 / lam[S[i - 1]])
            Q[i] = max([Q[i - 1] + (r[S[i - 1]] * dt), 0])
            if p <= dt:
                S[i] = (S[i - 1] + 1) % 2
            else:
                S[i] = S[i - 1]

        W = W + (Q <= x)
        F1 = F1 + ((Q <= x) * (S == 0))
        F2 = F2 + ((Q <= x) * (S == 1))

    W = W / iter
    F1 = F1 / iter
    F2 = F2 / iter

    return W.T, F1.T, F2.T


import numpy as np

Lambda_est = [1.3 ,0.4]
R =[ -1 , 4]
x_fix = 1
dt = 0.05
t_n = 20
t = range(0, int(t_n/dt))
t = np.array(t) * np.array([dt])
t = np.ndarray.tolist(t)

F , W_1 , W_2  = FQ( Lambda_est[0] , Lambda_est[1] , R[0] , R[1] , x_fix , 'o' , 1 , t_n , dt)
F_PS , W_1PS , W_2PS = FQ( Lambda_est[0] , Lambda_est[1] , R[0] , R[1], x_fix ,'p' , 1 , t_n , dt )
F_AS , W_1AS , W_2AS = FQ( Lambda_est[0] , Lambda_est[1] , R[0] , R[1], x_fix ,'a' , 1 , t_n , dt )
F_C , W_1C , W_2C = FQ( Lambda_est[0] , Lambda_est[1] , R[0] , R[1] , x_fix , 'c' , 1 , t_n , dt )
F_M , W_1M , W_2M = MonteCarloSimulationFQ( Lambda_est[0] , Lambda_est[1] , R[0] , R[1] , x_fix , t_n , dt )

import matplotlib.pyplot as plt
# Senario I
# F vs PS

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('Comparison of Senario I')
line1, = ax1.plot(t , F, 'b', alpha=0.3)
line2, = ax1.plot(t , F_PS, 'r', alpha=0.3 )
ax1.set_title('(a) Prob (Q(t) \leq x) ')
ax1.legend ([line1, line2],['F(t,x)','F(t,x) Approximated '])
# axis ([0 ,20 ,0 ,1])


line1, =ax2.plot (t , W_1  ,'b')
line2, =ax2.plot (t , W_2  ,'r')
line3, =ax2.plot (t , W_1PS ,'y')
line4, =ax2.plot (t , W_2PS ,'g')
ax2.set_title ('(b) W ^1(t,x) and W^2(t,x)')
ax2.legend ((line1, line2, line3, line4), ('W ^1(t,x)','W ^2(t,x)','W^1(t,x) Approximated ','W^2(t,x) Approximated'))
# axis ([0 ,20 ,0 ,1])


ax3.plot (t , abs ( F - F_PS ) )
ax3.set_title ('(c) Absolute Difference of F(t,x)')
# axis ([0 ,20])

ax4.plot (t ,( abs ( F_PS - F ) / F  ) )
ax4.set_title ('(d) Relative Difference of F(t,x)')

# Distance
L1_I = sum ( abs ( F  - F_PS  ) * dt )
L2_I =  ( sum ((( F  - F_PS  )**2) * dt ) )**(0.5)
Linf_I = np.max(np.array( abs ( F  - F_PS  ) * dt) )
Lmean_I = np.mean(np.array( abs ( F  - F_PS  ) * dt) )
L_I =[ L1_I, L2_I, Linf_I, Lmean_I ]


# Scenario 2
# F vs. AS


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('Comparison of Senario II')
line1, = ax1.plot(t , F,)
line2, = ax1.plot(t , F_AS  )
ax1.set_title('(a) Prob (Q(t) \leq x) ')
ax1.legend ([line1, line2],['F(t,x)','F(t,x) Approximated '])
# axis ([0 ,20 ,0 ,1])


line1, =ax2.plot (t , W_1  ,'b')
line2, =ax2.plot (t , W_2  ,'r')
line3, =ax2.plot (t , W_1AS ,'g')
line4, =ax2.plot (t , W_2AS ,'y')
ax2.set_title ('(b) W ^1(t,x) and W^2(t,x)')
ax2.legend ((line1, line2, line3, line4), ('W ^1(t,x)','W ^2(t,x)','W^1(t,x) Approximated ','W^2(t,x) Approximated'))
# axis ([0 ,20 ,0 ,1])


ax3.plot (t , abs ( F - F_AS ) )
ax3.set_title ('(c) Absolute Difference of F(t,x)')
# axis ([0 ,20])

ax4.plot (t ,( abs ( F_AS - F ) / F  ) )
ax4.set_title ('(d) Relative Difference of F(t,x)')

# Distance
L1_II = sum ( abs ( F  - F_AS  ) * dt )
L2_II =  ( sum ((( F  - F_AS  )**2) * dt ) )**(0.5)
Linf_II = np.max(np.array( abs ( F  - F_AS  ) * dt) )
Lmean_II = np.mean(np.array( abs ( F  - F_AS  ) * dt) )
L_II =[ L1_II, L2_II, Linf_II, Lmean_II ]


# Scenario 3
# F vs. Combination



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('Comparison of Senario III')
line1, = ax1.plot(t , F,)
line2, = ax1.plot(t , F_C  )
ax1.set_title('(a) Prob (Q(t) \leq x) ')
ax1.legend ([line1, line2],['F(t,x)','F(t,x) Approximated '])
# axis ([0 ,20 ,0 ,1])


line1, =ax2.plot (t , W_1  ,'b')
line2, =ax2.plot (t , W_2  ,'r')
line3, =ax2.plot (t , W_1C ,'g')
line4, =ax2.plot (t , W_2C ,'y')
ax2.set_title ('(b) W ^1(t,x) and W^2(t,x)')
ax2.legend ((line1, line2, line3, line4), ('W ^1(t,x)','W ^2(t,x)','W^1(t,x) Approximated ','W^2(t,x) Approximated'))
# axis ([0 ,20 ,0 ,1])


ax3.plot (t , abs ( F - F_C ) )
ax3.set_title ('(c) Absolute Difference of F(t,x)')
# axis ([0 ,20])

ax4.plot (t ,( abs ( F_C - F ) / F  ) )
ax4.set_title ('(d) Relative Difference of F(t,x)')

# Distance
L1_III = sum ( abs ( F  - F_C  ) * dt )
L2_III =  ( sum ((( F  - F_C  )**2) * dt ) )**(0.5)
Linf_III = np.max(np.array( abs ( F  - F_C  ) * dt) )
Lmean_III = np.mean(np.array( abs ( F  - F_C  ) * dt) )
L_III =[ L1_III, L2_III, Linf_III, Lmean_III ]

# Scenario IV
# F vs. MC
F_M_copy = F_M[1:]
W_1M_copy = W_1M[1:]
W_2M_copy = W_2M[1:]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('Comparison of Senario IV')
line1, = ax1.plot(t , F,)
line2, = ax1.plot(t , F_M_copy  )
ax1.set_title('(a) Prob (Q(t) \leq x) ')
ax1.legend ([line1, line2],['F(t,x)','F(t,x) Simulated '])
# axis ([0 ,20 ,0 ,1])


line1, =ax2.plot (t , W_1  ,'b')
line2, =ax2.plot (t , W_2  ,'r')
line3, =ax2.plot (t , W_1M_copy ,'g')
line4, =ax2.plot (t , W_2M_copy ,'y')
ax2.set_title ('(b) W ^1(t,x) and W^2(t,x)')
ax2.legend ((line1, line2, line3, line4), ('W ^1(t,x)','W ^2(t,x)','W^1(t,x) Simulated ','W^2(t,x) Simulated'))
# axis ([0 ,20 ,0 ,1])


ax3.plot (t , abs ( F - F_M_copy ) )
ax3.set_title ('(c) Absolute Difference of F(t,x)')
# axis ([0 ,20])

ax4.plot (t ,( abs ( F_M_copy - F ) / F  ) )
ax4.set_title ('(d) Relative Difference of F(t,x)')

# Distance
L1_IV = sum ( abs ( F  - F_M_copy  ) * dt )
L2_IV =  ( sum ((( F  - F_M_copy  )**2) * dt ) )**(0.5)
Linf_IV = np.max(np.array( abs ( F  - F_M_copy  ) * dt) )
Lmean_IV = np.mean(np.array( abs ( F  - F_M_copy  ) * dt) )
L_IV =[ L1_IV, L2_IV, Linf_IV, Lmean_IV ]


# Scenario V
# MC vs. Combination 2

F_M_copy = F_M[1:]
W_1M_copy = W_1M[1:]
W_2M_copy = W_2M[1:]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('Comparison of Senario V')
line1, = ax1.plot(t , F_C)
line2, = ax1.plot(t , F_M_copy  )
ax1.set_title('(a) Prob (Q(t) \leq x) ')
ax1.legend ([line1, line2],['F(t,x) Approximated (Combination)','F(t,x) Simulated '])
# axis ([0 ,20 ,0 ,1])


line1, =ax2.plot (t , W_1C  ,'b')
line2, =ax2.plot (t , W_2C  ,'r')
line3, =ax2.plot (t , W_1M_copy ,'g')
line4, =ax2.plot (t , W_2M_copy ,'y')
ax2.set_title ('(b) W ^1(t,x) and W^2(t,x)')
ax2.legend ((line1, line2, line3, line4), ('W ^1(t,x) Approximated (Combination)','W ^2(t,x) Approximated (Combination)','W^1(t,x) Simulated ','W^2(t,x) Simulated'))
# axis ([0 ,20 ,0 ,1])


ax3.plot (t , abs ( F_C - F_M_copy ) )
ax3.set_title ('(c) Absolute Difference')
# axis ([0 ,20])

ax4.plot (t ,( abs ( F_M_copy - F_C ) / F_C  ) )
ax4.set_title ('(d) Relative Difference')

# Distance
L1_V = sum ( abs ( F_C  - F_M_copy  ) * dt )
L2_V =  ( sum ((( F_C  - F_M_copy  )**2) * dt ) )**(0.5)
Linf_V = np.max(np.array( abs ( F_C  - F_M_copy  ) * dt) )
Lmean_V = np.mean(np.array( abs ( F_C  - F_M_copy  ) * dt) )
L_V =[ L1_V, L2_V, Linf_V, Lmean_V ]

## LOSS TABLE
print("L1 loss, L2 loss, Linf loss, L_mean loss")
print("Scenario I")
print(L_I)
print("Scenario II")
print(L_II)
print("Scenario III")
print(L_III)
print("Scenario IV")
print(L_IV)
print("Scenario V")
print(L_V)