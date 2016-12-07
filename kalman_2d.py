#%%

import numpy
import matplotlib.pyplot as plt

xs = []
truth = []

b = numpy.array([1.0, 1.0])
a = numpy.array([0.2, 0.8])

dt = 1.0 / 30.0

t = 0.0
T = 60 * dt

D = 0.2

while t <= T:
    xs.append(a * t + b + D * numpy.random.randn(b.shape[0]))
    truth.append(a)# * t + b)
    
    t += dt
    
xs = numpy.array(xs)
truth = numpy.array(truth)

#plt.plot(truth[:, 0], truth[:, 1], 'r')
#plt.plot(xs[:, 0], xs[:, 1], 'b*')
plt.plot(truth, 'r')
plt.plot(xs, 'b*')

plt.legend(['Reference', 'Data'], loc = "upper left")
plt.show()
#%%

xh = b.copy()
s = 1.0

S = 10
ddt = dt / S


xhs = []
ss = []

#while t <= T:

    #dx = F * x * dt + C * numpy.random.randn() * numpy.sqrt(dt)
for i in range(1, len(xs)):
    dz = xs[i] - xs[i - 1]
    #dz = G * x * dt + D * numpy.random.randn() * numpy.sqrt(dt)


    for i in range(S):
        s = s + -ddt * (s**2 / D**2)
        #xh += 

    xh += (s / D**2) * (dz - xh * dt)
    #print xh, (s / D**2) * xh, (s / D**2) * dz
    #xh += dxh

    #print dxh

    xhs.append(xh.copy())
    ss.append(s)

    t += dt
    
xhs = numpy.array(xhs)

#plt.plot(xs, 'r')
#plt.plot(truth[:, 0], truth[:, 1], 'r')
#plt.plot(xs[:, 0], xs[:, 1], 'b--*')
#plt.plot(xhs[:, 0], xhs[:, 1], 'g--o')
plt.plot(truth, 'r')
plt.plot(xs, 'b--*')
plt.plot(xhs, 'g--o')
plt.legend(['Reference', 'Data', 'Estimated'], loc = "upper left")
plt.show()

#plt.plot(zs, 'b')
#plt.plot(ss, 'k')