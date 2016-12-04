#%%

import numpy
import matplotlib.pyplot as plt

N = 2
Dm = 2

#F = 0.0
C = 0.01

#G = 1.0
D = 0.01

dt = 0.01

q = numpy.sqrt(0.001)

r = numpy.linspace(0, 0.5)
#plt.plot(r, q**2 / r**2)
#plt.ylim(0.0, 10.0)
#plt.show()

fs = 5.0
fo = 0.2

sm = 1 / (1 + numpy.exp(fs * (r - fo)))
plt.plot(r, sm)
plt.show()

def fF(x, p = False):
    N = x.shape[0]

    f = numpy.zeros(x.shape)
    F = numpy.zeros((x.shape[0], x.shape[1], x.shape[0], x.shape[1]))

    for i in range(N):
        F[i] = 0.0

        for j in range(N):
            if j == i:
                continue

            r2 = (x[i] - x[j]).dot(x[i] - x[j])

            r = numpy.sqrt(r2)

            #if r > 2 * fo:
            #    continue

            n = (x[j] - x[i]) / r

            fij = 1 / (1 + numpy.exp(fs * (r - fo)))

            f[i] += -fij * n

            #print numpy.linalg.norm(n)

            fpp = fs * fij * (fij - 1)

            dndx = (numpy.eye(Dm) - numpy.outer(n, n)) / r

            #for ii in range(Dm):
            #    for jj in range(Dm):
            #        F[i, ii, i, jj] += -fpp * n[ii] * -(x[j, jj] - x[i, jj]) / r + -fij * dndx[ii, ii]
            #        F[i, ii, j, jj] += -fpp * n[ii] * (x[j, jj] - x[i, jj]) / r + -fij * dndx[ii, jj]
            F[i, :, i, :] += numpy.outer(-fpp * n, -(x[j] - x[i]) / r) + fij * dndx
            F[i, :, j, :] += numpy.outer(-fpp * n, (x[j] - x[i]) / r) + -fij * dndx

            #if p:
            #    print numpy.outer(n, n)
            #    print 'dndx', dndx
            #    print F[i, :, i, :], i
            #    print (x[i] - x[j]) / r, x[i], x[j], i, j, fpp, fij
            #print i, j, r, fs * fij * (1 - fij), fij, fs
            #print 'hi', (fs * fij * (1 - fij) * (x[i] - x[j]) / r)
            #print -(fs * fij * (1 - fij) * (x[i] - x[j]) / r)

    return f, F

#f, F = fF(x)
#
#print ""
#
#Fp = numpy.zeros(F.shape)
#for i in range(x.shape[0]):
#    for j in range(x.shape[1]):
#        x1 = x.copy()
#        x1[i, j] += 0.000001
#
#        f1, F1 = fF(x1)
#
#        print ""
#
#        Fp[:, :, i, j] = (f1 - f) / (x1[i, j] - x[i, j])
#
#f, F = fF(x, True)
#
#print Fp[:, :, :, :]
##print Fp.reshape(N, N)
#print ""
#print F[:, :, :, :]
##print F.reshape(N, N)
x0 = numpy.random.rand(N, Dm) * 0.1 + 0.5
z0 = numpy.zeros((N, Dm))
xh0 = x0.copy()#numpy.random.rand(N, Dm)

s0 = numpy.eye(N * Dm).reshape((N, Dm, N, Dm)) * 0.01

#%%

def fFt(x):
    F = numpy.eye(N * Dm).reshape(s.shape) * 0.1

    return numpy.einsum('ijkl, kl', F, x), F

x = x0.copy()
z = z0.copy()
xh = xh0.copy()
s = s0.copy()

C = 0.1
D = 0.1

t = 0.0
dt = 0.001
T = 1000 * dt

xhs = []
zs = []
ss = [s.copy()]
xs = [x.copy()]
xhs = [xh.copy()]


while t <= T:
    #f, F = fF2(x)
    #fh, Fh = fF2(xh)

    f, F = fF(x)
    fh, Fh = fF(xh)

    dx = f * dt + C * numpy.random.randn(*x.shape) * numpy.sqrt(dt)
    dz = f * dt + D * numpy.random.randn(*x.shape) * numpy.sqrt(dt)

    #s = s + dt * (2 * F * s - (G**2 / D**2) * s**2 + C**2)

    SGDD = numpy.einsum('ijkl, mnkl', s, Fh) / (D**2)
    SGDDG = numpy.einsum('ijkl, klmn', SGDD, Fh)
    SGDDGS = numpy.einsum('ijkl, klmn', SGDDG, s)
    dxh = dt * numpy.einsum('ijkl, kl', Fh - SGDDG, xh) + numpy.einsum('ijkl, kl', SGDD, dz)#

    #print dxh#fh, numpy.einsum('ijkl, kl', Fh - SGDDG, xh)#dxh,  , numpy.einsum('ijkl, kl', SGDD, dz)
    s = s + dt * (numpy.einsum('ijkl, klmn', Fh, s) + numpy.einsum('ijkl, mnkl', s, Fh) - SGDDGS + (numpy.eye(N * Dm) * C**2).reshape(s.shape))

    x += dx
    xh += dxh
#
#    for n in range(N):
#        if x[n][0] < 0:
#            x[n][0] *= -1
#
#        if x[n][1] < 0:
#            x[n][1] *= -1
#
#        if x[n][0] > 1.0:
#            x[n][0] = 1.0 - (x[n][0] - 1.0)
#
#        if x[n][1] > 1.0:
#            x[n][1] = 1.0 - (x[n][1] - 1.0)
    #z += dz
    #xh += dxh

    #print dxh

    xs.append(x.copy())
    xhs.append(xh.copy())
    #zs.append(z)
    ss.append(s)

    t += dt

xs = numpy.array(xs)
xhs = numpy.array(xhs)
ss = numpy.array(ss)

for i in range(x.shape[0]):
    plt.plot(xs[:, i, 0], xs[:, i, 1], 'b')
    #plt.plot(xs[:, i, 1], 'b')
#plt.xlim((0.0, 1.0))
#plt.ylim((0.0, 1.0))

for i in range(xh.shape[0]):
    print 'hi'
    plt.plot(xhs[:, i, 0], xhs[:, i, 1], 'r')
    #plt.plot(xhs[:, i, 1], 'r')
#plt.xlim((-1.0, 2.0))
#plt.ylim((-1.0, 2.0))
plt.show()
#plt.plot(xhs, 'g')

#plt.plot(zs, 'b')
for i in range(N):
    for j in range(Dm):
        plt.plot(ss[:, i, j, i, j], 'k')