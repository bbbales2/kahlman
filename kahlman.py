#%%

import numpy
import matplotlib.pyplot as plt

x = 1.0
xh = 0.5
z = 0.0
s = 1.0

F = 0.0
C = 0.1

G = 1.0
D = 0.5

dt = 0.01

t = 0.0
T = 2000 * dt

xs = []
xhs = []
zs = []
ss = []

while t <= T:
    F = 0.0#numpy.sin(t) / 2.0

    dx = F * x * dt + C * numpy.random.randn() * numpy.sqrt(dt)
    dz = G * x * dt + D * numpy.random.randn() * numpy.sqrt(dt)

    dxh = (F - G**2 * s / D**2) * xh * dt + (G * s / D**2) * dz

    s = s + dt * (2 * F * s - (G**2 / D**2) * s**2 + C**2)

    x += dx
    z += dz
    xh += dxh

    #print dxh

    xs.append(x)
    zs.append(z)
    xhs.append(xh)
    ss.append(s)

    t += dt

plt.plot(xs, 'r')
plt.plot(xhs, 'g')

#plt.plot(zs, 'b')
plt.plot(ss, 'k')