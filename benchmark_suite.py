"""Small sanity suite; replace/extend with official CEC functions for papers."""

import numpy as np


def sphere(x): return float(np.sum(x**2))
def rosenbrock(x): return float(np.sum(100*(x[1:]-x[:-1]**2)**2+(x[:-1]-1)**2))
def rastrigin(x): return float(10*x.size+np.sum(x**2-10*np.cos(2*np.pi*x)))
def ackley(x):
    return float(-20*np.exp(-.2*np.sqrt(np.mean(x**2)))-np.exp(np.mean(np.cos(2*np.pi*x)))+20+np.e)
def griewank(x): return float(np.sum(x**2)/4000-np.prod(np.cos(x/np.sqrt(np.arange(1,x.size+1))))+1)
def schwefel(x): return float(418.9828872724338*x.size-np.sum(x*np.sin(np.sqrt(np.abs(x)))))
def zakharov(x):
    s=np.sum(.5*np.arange(1,x.size+1)*x); return float(np.sum(x**2)+s**2+s**4)


FUNCTIONS = {
    "sphere": (sphere, -100, 100), "rosenbrock": (rosenbrock, -5, 10),
    "rastrigin": (rastrigin, -5.12, 5.12), "ackley": (ackley, -32.768, 32.768),
    "griewank": (griewank, -600, 600), "schwefel": (schwefel, -500, 500),
    "zakharov": (zakharov, -5, 10),
}
