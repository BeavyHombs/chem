import numpy as np
from numpy import linalg as la
from numpy import random as rnd
import matplotlib.pyplot as plt
import math
from numba import jit
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

@jit(nogil=True)
def simulate(t):
  # use local variables because stack accesses are faster than reading from global
  l, n, m, k, pi = .005, 4194304, 1.008 / (6.02214e23), 1.380649e-23, 3.141592653589793
  
  r = np.zeros((n, 3))
  v = np.zeros((n, 3))
  
  for i in range(n):
    theta = rnd.uniform(-pi, pi)
    phi = rnd.uniform(-pi / 2, pi / 2)
    x, y, z = math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi), math.sin(phi)
    v[i] = np.array([x, y, z]) / math.sqrt(x**2 + y**2 + z**2) * rnd.normal(0, math.sqrt(k*t/m))
    r[i] = np.array([rnd.uniform(0, l), rnd.uniform(0, l), rnd.uniform(0, l)])
  
  avg_vel = np.sum(v, axis=0) / n
  for i in range(n):
    v[i] -= avg_vel
  
  total_v = 0
  for vi in v:
    total_v += np.sum(np.square(vi))
  init_t = m / k / n * total_v / 3

  scale = math.sqrt(t / init_t)
  for i in range(n):
    v[i] *= scale

  return loop(r, v, m, l, t)


@jit(nogil=True)
def loop(r, v, m, l, t):
  dt = 5e-6
  n = 0
  total = 0
  n_iters = 1024
  surface_area = 6*l*l

  for _ in range(n_iters):
    j1, j2, j3 = 0, 0, 0
    for i in range(len(r)):
      r[i] += v[i] * dt
      a0 = r[i][0] <= 0 or r[i][0] >= l
      a1 = r[i][1] <= 0 or r[i][1] >= l
      a2 = r[i][2] <= 0 or r[i][2] >= l
      j1 += 2 * m * math.fabs(v[i][0]) * a0
      j2 += 2 * m * math.fabs(v[i][1]) * a1
      j3 += 2 * m * math.fabs(v[i][2]) * a2
      v[i][0] *= a0 * -2 + 1
      v[i][1] *= a1 * -2 + 1
      v[i][2] *= a2 * -2 + 1
  
    p = (j1 + j2 + j3) / surface_area / dt
    total += p

  return t, total / n_iters


def main():
  tpe = ThreadPoolExecutor(max_workers=16)
  result_iter = tpe.map(simulate, [30 + 15 * i for i in range(40)])
  result_map : dict[int, float] = {}
  with open("results.txt", "w") as f:
    for it in result_iter:
      result_map[it[0]] = it[1]
      f.write(repr(it) + '\n')

  fig = plt.figure()
  x = np.array([4194304 / 6.02214076e23 * temp for temp in result_map.keys()]).reshape(-1, 1)
  y = np.array([0.000000125 * pressure for pressure in result_map.values()]).reshape(-1, 1)
  reg = LinearRegression().fit(x, y)
  plt.plot(x, y, 'bo')
  plt.xlabel("nT (mol*K)")
  plt.ylabel("PV (Pa*m^3)")
  plt.suptitle("PV vs. nT", y=1.05, fontsize=18)
  plt.title(f"y = {reg.coef_}x + {reg.intercept_}, R^2 = {reg.score(x, y)}", fontsize=10)
  plt.show()


if __name__ == "__main__":
  main()
