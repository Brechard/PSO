import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('-plot', action='store', dest='plot', help='Plot every x iteration')
parser.add_argument('-p', action='store', dest='particles', help='Number of particles')
parser.add_argument('-i', action='store', dest='iterations', help='Number of iterations')

settings = parser.parse_args()
ppp = 0


def plot_cost_function(fig=plt.figure(), option=0):
	X = np.linspace(-5, 5, 100)
	Y = np.linspace(-5, 5, 100)
	X, Y = np.meshgrid(X, Y)
	if option == 0:
		Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
	else:
		Z = (1 - Y) ** 2 + 2 * (Y - X ** 2) ** 2
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.4, cmap=cm.nipy_spectral, linewidth=0.08)
	# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.nipy_spectral, linewidth=0.08)
	ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
	ax.contourf(X, Y, Z, zdir='x', offset=5, cmap=cm.coolwarm)
	ax.contourf(X, Y, Z, zdir='y', offset=5, cmap=cm.coolwarm)


# plt.savefig('rastrigin_graph.png')
# plt.show()


if settings.particles:
	n_particles = int(settings.particles)
else:
	while True:
		try:
			n_particles = int(input("Input number of particles: "))
			break
		except ValueError:
			print("The number of particles must be an integer")


def calculate_cost(func_cost=0):
	if func_cost == 0:
		return (pos[:, 0] ** 2 - 10 * np.cos(2 * np.pi * pos[:, 0])) + (
				pos[:, 1] ** 2 - 10 * np.cos(2 * np.pi * pos[:, 1])) + 20
	else:
		return (1 - pos[:, 1]) ** 2 + 2 * (pos[:, 1] - pos[:, 0] ** 2) ** 2


def update_best():
	global best_swarm_pos, best_swarm_cost
	for part in range(n_particles):
		if cost[part] < best_cost[part]:
			best_pos[part, :] = pos[part, :]
			best_cost[part] = cost[part]

			if best_swarm_cost is None or cost[part] < best_swarm_cost:
				best_swarm_pos = pos[part, :]
				best_swarm_cost = cost[part]
				print("\033[94m New best particle in pos:", best_swarm_pos, "with a cost of:", best_swarm_cost,
					  "\033[0m")


pos = np.random.rand(n_particles, 2) * 10 - 5
vel = np.random.rand(n_particles, 2) * 10 - 5
boundaries_min = np.transpose([np.full(n_particles, -5), np.full(n_particles, -5)])
boundaries_max = np.transpose([np.full(n_particles, 5), np.full(n_particles, 5)])
best_pos = pos
cost = calculate_cost()
best_cost = cost
best_swarm_cost = np.min(best_cost)
best_swarm_pos = pos[np.argmin(best_cost)]
a = 0.9
b = 2
c = 2


def update(n_times=10000, option=0):
	global best_swarm_pos, best_swarm_cost, cost, vel, pos, a

	factor = 0.5 / n_times
	for i in range(n_times):
		rp = np.random.rand(n_particles, 2)
		rg = np.random.rand(n_particles, 2)
		cost = calculate_cost(option)
		update_best()
		vel = a * vel + b * rp * (best_pos - pos) + c * rg * (best_swarm_pos - pos)
		pos = pos + vel
		for part in range(len(pos)):
			for j in range(len(pos[part])):
				if pos[part, j] > 5:
					pos[part, j] = 5
				elif pos[part, j] < -5:
					pos[part, j] = -5
		if a > 0.4:
			a -= factor
		# print(pos)
		if settings.plot and i % int(settings.plot) == 0:
			t = "\033[91m Iteration :" + str(i) + ". Worst particle is at " + str(
				pos[np.argmax(cost)]) + " with a cost of: " + str(
				np.max(cost)) + "\033[0m"
			print(t)
			plot_particles(option, "Iteration " + str(i))
		if np.max(cost) < 0.01:
			t = "\033[91m Finished in iteration :" + str(i) + ". Worst particle is at " + str(
				pos[np.argmax(cost)]) + " with a cost of: " + str(
				np.max(cost)) + "\033[0m"
			print(t)
			break


def plot_particles(option, title="Particles", cost_funct=True, save_fig=False):
	global ppp
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(pos[:, 0], pos[:, 1], cost, marker='v', c='#000000')
	ax.scatter(pos[:, 0], pos[:, 1], [0 for i in range(len(pos))], marker='v')
	plt.title(title)
	plt.ylim(-5, 5)
	plt.xlim(-5, 5)
	# plt.savefig('particles_graph' + str(ppp) + '.png')
	if cost_funct:
		plot_cost_function(fig, option)
	# ax.view_init(90, 0)
	if save_fig:
		plt.savefig('cost_funct_graph2' + str(ppp) + '.png')
	plt.show()
	ppp += 1


# plot_particles()
if settings.iterations:
	update(int(settings.iterations), 1)
else:
	update(1000)

# plot_particles()
