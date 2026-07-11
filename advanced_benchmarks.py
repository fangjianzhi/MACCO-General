"""Deterministic shifted/rotated benchmarks for structural validation."""

import numpy as np


def _rotation(dim, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    q *= np.sign(np.diag(r))[None, :]
    return q


def make_advanced_suite(dim, seed=20260711):
    q = _rotation(dim, seed + dim)
    shift = np.random.default_rng(seed + 2 * dim).uniform(-0.35, 0.35, dim)

    def transform(x, scale): return q @ (x / scale - shift)
    def rotated_sphere(x):
        z=transform(x,100); return float(np.sum(z**2))
    def rotated_ellipsoid(x):
        z=transform(x,100); return float(np.sum((1e6**(np.arange(dim)/max(dim-1,1)))*z**2))
    def rotated_rosenbrock(x):
        z=transform(x,5)+1; return float(np.sum(100*(z[1:]-z[:-1]**2)**2+(z[:-1]-1)**2))
    def rotated_rastrigin(x):
        z=transform(x,5.12); return float(10*dim+np.sum(z**2-10*np.cos(2*np.pi*z)))
    def rotated_ackley(x):
        z=transform(x,32.768); return float(-20*np.exp(-.2*np.sqrt(np.mean(z**2)))-np.exp(np.mean(np.cos(2*np.pi*z)))+20+np.e)
    def bent_cigar(x):
        z=transform(x,100); return float(z[0]**2+1e6*np.sum(z[1:]**2))
    return {
        "rot_sphere": (rotated_sphere,-100,100),
        "rot_ellipsoid": (rotated_ellipsoid,-100,100),
        "rot_rosenbrock": (rotated_rosenbrock,-5,5),
        "rot_rastrigin": (rotated_rastrigin,-5.12,5.12),
        "rot_ackley": (rotated_ackley,-32.768,32.768),
        "bent_cigar": (bent_cigar,-100,100),
    }
