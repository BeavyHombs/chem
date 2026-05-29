# Ideal Gas 3D Molecular Dynamics Simulation

**By: Michael L., Nikko L., Noah Y.**

## Goal

The goal of this project was to create a 3D molecular dynamics simulation of an ideal gas and use it to verify the Ideal Gas Law:

\[
PV = nRT
\]

The simulation models gas particles in a cubic container, calculates the pressure from wall collisions, and compares \(PV\) against \(nT\).

## Simulation Setup

- Container: cube with side length \(L = 0.005\) m
- Volume: \(V = L^3 = 1.25 \times 10^{-7}\) m³
- Number of molecules for pressure calculation: \(N = 65{,}536\)
- Number of molecules for animation: \(N = 256\)
- Particle mass: based on hydrogen, using \(1.008\) g/mol divided by Avogadro’s number
- Temperature range: 30 K to 615 K
- Gravity and inter-particle collisions were ignored
- Wall collisions were treated as perfectly elastic

## How the Simulation Works

Each molecule starts at a random position inside the cube. Its velocity components are randomly generated from a normal distribution based on the target temperature. The average velocity is then subtracted so the gas does not drift in one overall direction. After that, the velocities are rescaled so the system’s kinetic energy matches the chosen temperature.

During the simulation, each particle moves according to:

\[
r_{new} = r_{old} + v \Delta t
\]

When a particle hits a wall, the velocity component perpendicular to that wall is reversed. The impulse from each wall collision is added up and used to calculate pressure:

\[
P = \frac{\sum \Delta p}{A \Delta t}
\]

where \(A = 6L^2\), the total surface area of the cube.

## Analysis

The program runs the simulation at multiple temperatures while keeping the number of particles and container volume constant. For each temperature, it calculates the pressure, then graphs \(PV\) against \(nT\).

According to the Ideal Gas Law:

\[
PV = R(nT)
\]

So the graph should be linear, with the slope representing the simulation’s estimate of the gas constant \(R\).

## Conclusion

The simulation showed an approximately linear relationship between \(PV\) and \(nT\), supporting the Ideal Gas Law. Small differences from the expected value are reasonable because the simulation uses discrete time steps and simplifies the gas by ignoring inter-particle collisions.