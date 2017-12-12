import networkx as nx
import numpy as np
from random import uniform, randint
import matplotlib.pyplot as plt

def createFileGraph(file):
	G = nx.read_edgelist(file, create_using = nx.Graph(), nodetype = int)
	return G

def updateGraph(G):
	G.add_nodes_from(G.nodes, r = 0)
	for node in G.nodes:
		G.add_nodes_from([node], p = round(uniform(0, 1), 2))
		G.add_nodes_from([node], q = round(uniform(0, 1), 2))
	return G

def meanMetrics(G):
	pMean = 0
	qMean = 0
	rMean = 0
	for node in G.nodes:
		pMean += G.node[node]['p']
		qMean += G.node[node]['q']
		rMean += G.node[node]['r']
	return [round(pMean/len(G.nodes), 2), round(qMean/len(G.nodes), 2)]

def noiseUG(epsilon = 0.01):
	result = (uniform(0, 1) * epsilon * 2) - epsilon
	return result

def resetReward(G):
	for node in G.nodes:
		G.node[node]['r'] = 0

def updateMetrics(G, updatedDict):
	for node in updatedDict:
		G.node[node]['p'] = updatedDict[node][0]
		G.node[node]['q'] = updatedDict[node][1]

def ultimatumGame(G, maxIterations = 1, epsilon = 0.01):
	updateGraph(G)
	evolve = []
	evolve.append(meanMetrics(G))
	for n in range(0, maxIterations):
		for node in G.nodes:
			for neighbor in G.neighbors(node):
				if G.node[node]['p'] >= G.node[neighbor]['q']:
					G.node[node]['r'] += 1 - G.node[node]['p']
					G.node[neighbor]['r'] += G.node[node]['p']

		updatedDict = {}
		for node in G.nodes:
			neighbors = list(G.neighbors(node))
			chosenNeighbor = neighbors[randint(0, len(neighbors) - 1)]

			if G.node[node]['r'] < G.node[chosenNeighbor]['r']:
				noiseNode = noiseUG(epsilon)
				if G.node[chosenNeighbor]['p'] + noiseNode > 0:
					newP = G.node[chosenNeighbor]['p'] + noiseNode
				else:
					newP = 0
				noiseNeighbor = noiseUG(epsilon)
				if G.node[chosenNeighbor]['q'] + noiseNeighbor > 0:
					newQ = G.node[chosenNeighbor]['q'] + noiseNeighbor
				else:
					newQ = 0
				updatedDict[node] = (newP, newQ)
		updateMetrics(G, updatedDict)
		evolve.append(meanMetrics(G))
		resetReward(G)
	return meanMetrics(G), evolve

def strategyVariation(graph, epsilon, maxIterations = 10):
	psum = 0
	qsum = 0
	for _ in range(0, maxIterations):
		psum += ultimatumGame(graph, epsilon = epsilon)[0][0]
		qsum += ultimatumGame(graph, epsilon = epsilon)[0][1]
	pmean = psum / maxIterations
	qmean = qsum / maxIterations
	return pmean, qmean

def epsilonVariation(graph):
	epsilonList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
	strategyList = []
	for epsilon in epsilonList:
		strategyMean = strategyVariation(graph, epsilon)
		strategyList.append(strategyMean)
	pValues = []
	qValues = []
	for strategy in strategyList:
		pValues.append(strategy[0])
		qValues.append(strategy[1])
	return pValues, qValues

def getEpsilonGraph(strategyList):
	epsilonList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
	plt.plot(epsilonList, strategyList[0], label = 'p Value')
	plt.plot(epsilonList, strategyList[1], label = 'q Value')
	plt.xlabel('Epsilon Value')
	plt.ylabel('p Value')
	plt.legend()
	plt.title('Epsilon Variation')
	plt.show()

def getTimeGraph(graph, maxIterations = 50):
	tVar = ultimatumGame(graph)[1]
	tAverage = tVar
	for _ in range(0, maxIterations - 1):
		tVar = ultimatumGame(graph)[1]
		for i in range(0, len(tVar)):
			tAverage[i][0] += tVar[i][0]
			tAverage[i][1] += tVar[i][1]

	for i in range(0, len(tAverage)):
		tAverage[i][0] /= maxIterations
		tAverage[i][1] /= maxIterations

	pOverTime = []
	qOverTime = []
	x = np.arange(len(tAverage))
	for strategy in tAverage:
		pOverTime.append(strategy[0])
		qOverTime.append(strategy[1])

	plt.plot(x, pOverTime, label = 'p Value')
	plt.plot(x, qOverTime, label = 'q Value')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Strategy')
	plt.legend()
	plt.title('Strategy Variation over Time')
	plt.show()
	

def main():
	#Create networks
	G1 = nx.random_regular_graph(99, 100) #Fully Connected
	G2 = nx.watts_strogatz_graph(100, 4, 0.05) #Ring
	G3 = nx.grid_2d_graph(10, 10) #Grid
	G4 = createFileGraph('edges_social.txt') #Real Social

	graphs = [G1, G2, G3]
	graphNames = ['Fully Connected Graph', 'Ring Graph', 'Grid Graph']

	#Calculate metrics
	for graphID in range(0, len(graphs)):
		print("----- Calculating metrics for", graphNames[graphID], '-----')
		epsilonVar = epsilonVariation(graphs[graphID])
		getEpsilonGraph(epsilonVar)
		getTimeGraph(graphs[graphID])

if __name__ == '__main__':
	main()