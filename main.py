import random
import math
import numpy as np
import queue
import argparse
from tqdm import trange

'''
Author:	Michel Conrado Cardoso Meneses
Date: First semester of 2018
License: MIT
'''

#-------------------ABSTRACTION-------------------

class Problem(object):

	def eval():
		print("Do it!")

class Solution(object):

	def __init__(self, dim, rg, pTweak = 0.7):
		self.value = []
		self.dim = dim
		self.rg = rg
		self.pTweak = pTweak
		
		for i in range(self.dim):
			self.value.append(random.uniform(self.rg[0],self.rg[1]))

	def tweak(self):
		print("Do it!")

class Pipeline(object):

	def __init__(self, prob, sol, it):
		self.prob = prob
		self.sol = sol
		self.it = it

	def run(self):
		print("Do it!")

#-------------------IMPLEMENTATION-------------------

class SphereProblem(Problem):

	def __init__(self):
		self.bias = 0

	def eval(self, sol):
		sum = 0
		for v in sol:
			sum = sum + v**2 + self.bias
		return sum

class SchwefelsProblem(Problem):

	def __init__(self):
		self.bias = 0

	def eval(self, sol):
		return max(map(abs, sol)) + self.bias

class RosenbrocksProblem(Problem):

	def __init__(self):
		self.bias = 0

	def eval(self, sol):
		sum = 0
		for i in range(len(sol) - 1):
			sum = sum + 100*(sol[i]**2 - sol[i+1])**2 + (sol[i] - 1)**2 + self.bias
		return sum

class RastringinsProblem(Problem):

	def __init__(self):
		self.bias = 0

	def eval(self, sol):
		sum = 0
		for v in sol:
			sum = sum + (v**2 - 10*math.cos(2*math.pi*v) + 10) + self.bias
		return sum

class SimpleSolution(Solution):

	def tweak(self):
		R = list(self.value)
		for i, r in enumerate(R):
			if random.random() < self.pTweak:
				R[i] = random.uniform(self.rg[0],self.rg[1])
		return R

class SmoothSolution(Solution):

	def __init__(self, dim, rg, pTweak, noiseRg):
		super(SmoothSolution, self).__init__(dim, rg, pTweak)
		self.noiseRg = noiseRg

	def tweak(self):
		R = list(self.value)
		for i, r in enumerate(R):
			if random.random() < self.pTweak:
				while True:
					n = random.uniform(-1*self.noiseRg, self.noiseRg)
					if R[i] + n > self.rg[0] and R[i] + n < self.rg[1]:
						break
				R[i] = R[i] + n
		return R

class CrazySolution(Solution):

	def tweak(self):
		R = list(self.value)
		for i, r in enumerate(R):
			if random.random() < random.random():
				R[i] = random.uniform(self.rg[0],self.rg[1])
		return R

class HCPipeline(Pipeline):

	def run(self):
		for i in trange(self.it):
			R = self.sol.tweak()
			if self.prob.eval(R) < self.prob.eval(self.sol.value):
				self.sol.value = R

class SAHCPipeline(Pipeline):

	def __init__(self, prob, sol, it, ngbrs):
		super(SAHCPipeline, self).__init__(prob, sol, it)
		self.ngbrs = ngbrs

	def run(self):
		for i in trange(self.it):
			R = self.sol.tweak()
			for j in range(self.ngbrs):
				W = self.sol.tweak()
				if self.prob.eval(W) < self.prob.eval(R):
					R = W
			if self.prob.eval(R) < self.prob.eval(self.sol.value):
				self.sol.value = R

class SAHCRPipeline(Pipeline):

	def __init__(self, prob, sol, it, ngbrs):
		super(SAHCRPipeline, self).__init__(prob, sol, it)
		self.ngbrs = ngbrs
	
	def run(self):
		Best = self.sol.value
		for i in trange(self.it):
			R = self.sol.tweak()
			for j in range(self.ngbrs):
				W = self.sol.tweak()
				if self.prob.eval(W) < self.prob.eval(R):
					R = W
			self.sol.value = R
			if self.prob.eval(self.sol.value) < self.prob.eval(Best):
				Best = self.sol.value
		self.sol.value = Best

class SAPipeline(Pipeline):

	def __init__(self, prob, sol, it, temp, step):
		super(SAPipeline, self).__init__(prob, sol, it)
		self.temp = temp
		self.step = step
	
	def run(self):
		Best = self.sol.value
		for i in trange(self.it):
			R = self.sol.tweak()
			if self.prob.eval(R) < self.prob.eval(self.sol.value) or random.random() < math.exp((self.prob.eval(self.sol.value) - self.prob.eval(R))/self.temp):
				self.sol.value = R
			if self.temp - self.step > 1:
				self.temp = self.temp - self.step	
			if self.prob.eval(self.sol.value) < self.prob.eval(Best):
				Best = self.sol.value
		self.sol.value = Best

class TBPipeline(Pipeline):

	def __init__(self, prob, sol, it, ngbrs, memSize):
		super(TBPipeline, self).__init__(prob, sol, it)
		self.ngbrs = ngbrs
		self.memSize = memSize
		self.tabu = queue.Queue()
	
	def run(self):
		Best = self.sol.value
		for i in trange(self.it):
			if self.tabu.qsize() > self.memSize:
				self.tabu.get()
			R = self.sol.tweak()
			TbR = self.createTabu(R, self.sol.value)
			for j in range(self.ngbrs):
				W = self.sol.tweak()
				TbW = self.createTabu(W, self.sol.value)
				if not TbW in self.tabu.queue and (self.prob.eval(W) < self.prob.eval(R) or TbR in self.tabu.queue):
					R = W
			if not TbR in self.tabu.queue:
				self.sol.value = R
				self.tabu.put(TbR)
			if self.prob.eval(self.sol.value) < self.prob.eval(Best):
				Best = self.sol.value
		self.sol.value = Best

	def createTabu(self, s1, s0):
		a0 = np.array(s0)
		a1 = np.array(s1)
		tb = a1 - a0
		tb[tb != 0] = 1
		return tb.tolist()

class ILSPipeline(Pipeline):

	def __init__(self, prob, sol, it, timeRg):
		super(ILSPipeline, self).__init__(prob, sol, it)
		self.timeRg = np.linspace(timeRg[0], timeRg[1], timeRg[2])
		self.H = SmoothSolution(self.sol.dim, self.sol.rg, self.sol.pTweak, self.sol.noiseRg * 1)
	
	def run(self):
		self.H.value = self.sol.value
		Best = self.sol.value
		for i in trange(self.it):
			time = int(random.choice(self.timeRg))
			for j in range(time):
				R = self.sol.tweak()
				if self.prob.eval(R) < self.prob.eval(self.sol.value):
					self.sol.value = R
			if self.prob.eval(self.sol.value) < self.prob.eval(Best):
				Best = self.sol.value
			self.H.value = self.newHomeBase(self.H.value, self.sol.value)
			self.sol.value = self.perturb(self.H)
		self.sol.value = Best

	def newHomeBase(self, H, S):
		if self.prob.eval(H) < 0.9*self.prob.eval(S):
			return H
		else:
			return S

	def perturb(self, H):
		return H.tweak()
#-----------------------MAIN------------------------

def solve(problem, alg):

	if problem == 'sphere':
		prob = SphereProblem()
	elif problem == 'schwefels':
		prob = SchwefelsProblem()
	elif problem == 'rosenbrocks':
		prob = RosenbrocksProblem()
	elif problem == 'rastringins':
		prob = RastringinsProblem()
	else:
		print("Error - unknown problem name!")
		quit()

	if alg == 'hc':
		sol = SmoothSolution(100, [-100,100], pTweak = 0.9, noiseRg=0.05)
		pip = HCPipeline(prob, sol, it=100000)
	elif alg == 'sahc':
		sol = SmoothSolution(100, [-100,100], pTweak = 0.9, noiseRg=0.05)
		pip = SAHCPipeline(prob, sol, it=33000, ngbrs=3)		
	elif alg == 'sahcr':
		sol = SmoothSolution(100, [-100,100], pTweak = 0.9, noiseRg=0.05)
		pip = SAHCRPipeline(prob, sol, it=33000, ngbrs=3)
	elif alg == 'sa':
		sol = SmoothSolution(100, [-100,100], pTweak = 0.9, noiseRg=0.05)
		pip = SAPipeline(prob, sol, it=100000, temp=0.0000001, step=0.000000001)		
	elif alg == 'tb':
		sol = SmoothSolution(100, [-100,100], pTweak = 0.9, noiseRg=0.05)
		pip = TBPipeline(prob, sol, it=33000, ngbrs=3, memSize=10)
	elif alg == 'ils':
		sol = SmoothSolution(100, [-100,100], pTweak = 0.9, noiseRg=0.02)
		pip = ILSPipeline(prob, sol, it=5000, timeRg=[10, 100, 10])
	else:
		print("Error - unknown algorithm name!")
		quit()

	pip.run()
	return prob, sol

parser = argparse.ArgumentParser(prog='Search', 
								 description='Run a search algorithm to solve an optimization problem.')
parser.add_argument("problem", help="The problem to be solved [sphere, schwefels, rosenbrocks or rastringins]")
parser.add_argument("algorithm", help="The algorithm used to solve the problem [hc, sahc, sahcr, sa, tb or ils]")
args = parser.parse_args()

prob, sol = solve(args.problem, args.algorithm)
print("Final cost: " + str(prob.eval(sol.value)))
