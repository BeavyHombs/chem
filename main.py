import numpy as np
from numpy import linalg as la
from numpy import random as rnd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from numba import jit
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor


def simulate(t):
  # use local variables because stack accesses are faster than reading from global
  l, n, m, k, pi = .005, 1048576, 1.008 / (6.022e23), 1.380649e-23, 3.1415926
  
  r = [None] * n
  v = [None] * n
  
  for i in range(n):
    theta = rnd.uniform(-pi, pi)
    phi = rnd.uniform(-pi/2, pi/2)
    x, y, z = math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi), math.sin(phi)
    v[i] = np.array([x, y, z]) / math.sqrt(x**2 + y**2 + z**2) * rnd.normal(0, math.sqrt(k*t/m))
    r[i] = np.array([rnd.uniform(0, l), rnd.uniform(0, l), rnd.uniform(0, l)])
  
  avg_vel = np.sum(v) / n
  v = [vi - avg_vel for vi in v]
  
  def cur_t():
    return .333333333 * m / k / n * np.sum([np.sum(np.square(vi)) for vi in v])
  
  init_t = cur_t()
  v = [vi * math.sqrt(t / init_t) for vi in v]

  return loop(r, v, m, l, t)


@jit(nogil=True)
def loop(r, v, m, l, t):
  dt = 1e-6
  time = 0
  n = 0
  total = 0
  n_iters = 128

  for _ in range(n_iters):
    j = 0
    for ri, vi in zip(r, v):
      j += (2 * m * math.fabs(vi[0]) * (ri[0] < 0 or ri[0] > l))
      j += (2 * m * math.fabs(vi[1]) * (ri[1] < 0 or ri[1] > l))
      j += (2 * m * math.fabs(vi[2]) * (ri[2] < 0 or ri[2] > l))

    for i, (ri, vi) in enumerate(zip(r, v)):
      r[i] = ri + vi * dt
      v[i] = np.array([vi[0] * ((ri[0] >= 0 and ri[0] <= l) * 2 - 1),
                       vi[1] * ((ri[1] >= 0 and ri[1] <= l) * 2 - 1),
                       vi[2] * ((ri[2] >= 0 and ri[2] <= l) * 2 - 1)])
  
    p = j / (6 * l * l) / dt
    total += p
  
    time += dt

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
  x = np.array([1048576 / 6.02214076e23 * temp for temp in result_map.keys()]).reshape(-1, 1)
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
