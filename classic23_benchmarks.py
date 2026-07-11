"""Source-compatible classic F1--F23 benchmark suite.

The formulas and bounds follow the supplied MATLAB ``Get_Functions_details.m``
files used by GWO, LEE, and the beaver optimizer projects.  F7 is deterministic
by default (quartic term only) so paired optimizer runs remain reproducible;
set ``noisy_f7=True`` to reproduce the MATLAB ``+ rand`` behavior.
"""

from __future__ import annotations

import numpy as np


def _u(x, a, k, m):
    return k * (x - a) ** m * (x > a) + k * (-x - a) ** m * (x < -a)


def make_classic23_suite(noisy_f7=False, noise_seed=0):
    rng = np.random.default_rng(noise_seed)

    def f1(x): return float(np.sum(x ** 2))
    def f2(x): return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))
    def f3(x): return float(np.sum(np.cumsum(x) ** 2))
    def f4(x): return float(np.max(np.abs(x)))
    def f5(x): return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))
    def f6(x): return float(np.sum(np.abs(x + .5) ** 2))
    def f7(x):
        value = np.sum(np.arange(1, x.size + 1) * x ** 4)
        return float(value + (rng.random() if noisy_f7 else 0.))
    def f8(x): return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))
    def f9(x): return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * x.size)
    def f10(x):
        return float(-20 * np.exp(-.2 * np.sqrt(np.mean(x ** 2)))
                     - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e)
    def f11(x):
        return float(np.sum(x ** 2) / 4000
                     - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1)
    def f12(x):
        y = 1 + (x + 1) / 4
        return float((np.pi / x.size) * (10 * np.sin(np.pi * y[0]) ** 2
                     + np.sum((y[:-1] - 1) ** 2
                              * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
                     + (y[-1] - 1) ** 2) + np.sum(_u(x, 10, 100, 4)))
    def f13(x):
        return float(.1 * (np.sin(3 * np.pi * x[0]) ** 2
                     + np.sum((x[:-1] - 1) ** 2
                              * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
                     + (x[-1] - 1) ** 2
                     * (1 + np.sin(2 * np.pi * x[-1]) ** 2))
                     + np.sum(_u(x, 5, 100, 4)))

    grid = np.array([[i, j] for j in (-32, -16, 0, 16, 32)
                     for i in (-32, -16, 0, 16, 32)], dtype=float).T
    def f14(x):
        b = np.sum((x[:, None] - grid) ** 6, axis=0)
        return float(1 / (1 / 500 + np.sum(1 / (np.arange(1, 26) + b))))

    ak = np.array([.1957, .1947, .1735, .16, .0844, .0627,
                   .0456, .0342, .0323, .0235, .0246])
    bk = 1 / np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    def f15(x):
        model = x[0] * (bk ** 2 + x[1] * bk) / (bk ** 2 + x[2] * bk + x[3])
        return float(np.sum((ak - model) ** 2))
    def f16(x):
        return float(4*x[0]**2 - 2.1*x[0]**4 + x[0]**6/3
                     + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4)
    def f17(x):
        return float((x[1] - 5.1*x[0]**2/(4*np.pi**2) + 5*x[0]/np.pi - 6)**2
                     + 10*(1 - 1/(8*np.pi))*np.cos(x[0]) + 10)
    def f18(x):
        a = 1 + (x[0]+x[1]+1)**2 * (19-14*x[0]+3*x[0]**2-14*x[1]
            +6*x[0]*x[1]+3*x[1]**2)
        b = 30 + (2*x[0]-3*x[1])**2 * (18-32*x[0]+12*x[0]**2+48*x[1]
            -36*x[0]*x[1]+27*x[1]**2)
        return float(a*b)

    ah3 = np.array([[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]])
    ph3 = np.array([[.3689,.117,.2673],[.4699,.4387,.747],
                    [.1091,.8732,.5547],[.03815,.5743,.8828]])
    ah6 = np.array([[10,3,17,3.5,1.7,8],[.05,10,17,.1,8,14],
                    [3,3.5,1.7,10,17,8],[17,8,.05,10,.1,14]])
    ph6 = np.array([[.1312,.1696,.5569,.0124,.8283,.5886],
                    [.2329,.4135,.8307,.3736,.1004,.9991],
                    [.2348,.1415,.3522,.2883,.3047,.6650],
                    [.4047,.8828,.8732,.5743,.1091,.0381]])
    ch = np.array([1, 1.2, 3, 3.2])
    def hartmann(x, a, p):
        return float(-np.sum(ch * np.exp(-np.sum(a * (x - p) ** 2, axis=1))))
    def f19(x): return hartmann(x, ah3, ph3)
    def f20(x): return hartmann(x, ah6, ph6)

    ash = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],
                    [3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],
                    [6,2,6,2],[7,3.6,7,3.6]], dtype=float)
    csh = np.array([.1,.2,.2,.4,.4,.6,.3,.7,.5,.5])
    def shekel(x, n):
        return float(-np.sum(1 / (np.sum((x - ash[:n]) ** 2, axis=1) + csh[:n])))
    def f21(x): return shekel(x, 5)
    def f22(x): return shekel(x, 7)
    def f23(x): return shekel(x, 10)

    funcs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,
             f14,f15,f16,f17,f18,f19,f20,f21,f22,f23]
    bounds = [(-100,100,30),(-10,10,30),(-100,100,30),(-100,100,30),
              (-30,30,30),(-100,100,30),(-1.28,1.28,30),(-500,500,30),
              (-5.12,5.12,30),(-32,32,30),(-600,600,30),(-50,50,30),
              (-50,50,30),(-65.536,65.536,2),(-5,5,4),(-5,5,2),
              (np.array([-5,0.]),np.array([10,15.]),2),(-2,2,2),
              (0,1,3),(0,1,6),(0,10,4),(0,10,4),(0,10,4)]
    return {f"F{i+1}": (funcs[i], *bounds[i]) for i in range(23)}


CLASSIC23 = make_classic23_suite()
