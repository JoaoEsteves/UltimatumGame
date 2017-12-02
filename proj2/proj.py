import networkx as nx
import numpy as np
from random import uniform, randint
import matplotlib.pyplot as plt

def createFileGraph(file):
	G = nx.read_edgelist(file, create_using = nx.Graph(), nodetype = int)
	return G

def createFCGraph():
	G =  nx.random_regular_graph(99, 100)
	return G

def createRingGraph():
	G =  nx.watts_strogatz_graph(100, 4, 0.05)
	return G

def createSocialGraph():
	G = nx.karate_club_graph()
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
	return round(pMean/len(G.nodes), 2)
	#return round(pMean/len(G.nodes), 2), round(qMean/len(G.nodes), 2)

def noiseUG(epsilon = 0.01):
	noise = [0, epsilon, - epsilon]
	result = np.random.choice(noise, 2, p = [0.8, 0.1, 0.1])
	return result[0], result[1]

def noiseUG2(epsilon = 0.01):
	result = (uniform(0, 1) * epsilon * 2) - epsilon
	return result

def resetReward(G):
	for node in G.nodes:
		G.node[node]['r'] = 0

def updateMetrics(G, updatedDict):
	for node in updatedDict:
		G.node[node]['p'] = updatedDict[node][0]
		G.node[node]['q'] = updatedDict[node][1]

def ultimatumGame(G, maxIterations = 1000, epsilon = 0.01):
	updateGraph(G)
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
				noiseNode = noiseUG2(epsilon)
				if G.node[chosenNeighbor]['p'] + noiseNode > 0:
					newP = G.node[chosenNeighbor]['p'] + noiseNode
				else:
					newP = 0
				noiseNeighbor = noiseUG2(epsilon)
				if G.node[chosenNeighbor]['q'] + noiseNeighbor > 0:
					newQ = G.node[chosenNeighbor]['q'] + noiseNeighbor
				else:
					newQ = 0
				updatedDict[node] = (newP, newQ)
		#print(meanMetrics(G))
		updateMetrics(G, updatedDict)
		resetReward(G)
	return meanMetrics(G)

def pVariation(graph, epsilon, maxIterations):
	psum = 0
	for _ in range(0, maxIterations):
		psum += ultimatumGame(graph, epsilon = epsilon)
	mean = psum / maxIterations
	return mean

def epsilonVariation(graph, maxIterations = 10):
	epsilonList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
	pList = []
	for epsilon in epsilonList:
		pMean = pVariation(graph, epsilon, maxIterations)
		pList.append(pMean)
	return pList

def getResults(graphs):
	graphsName = ['FCGraph', 'RingGraph', 'SocialGraph']
	graphsDict = {}
	for i in range(0, len(graphsName)):
		print(graphsName[i])
		graphsDict[graphsName[i]] = epsilonVariation(graphs[i])
	return graphsDict

def getGraphs(graphsDict):
	epsilonList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
	plt.plot(epsilonList, graphsDict['FCGraph'], label = 'Fully Connected Graph')
	plt.plot(epsilonList, graphsDict['RingGraph'], label = 'Ring Graph')
	plt.plot(epsilonList, graphsDict['SocialGraph'], label = 'Social Graph')
	plt.xlabel('Epsilon Value')
	plt.ylabel('p Value')
	plt.legend()
	plt.title('Epsilon Variation')
	plt.show()

'''
def epsilonVariation(graphs):
	epsilonList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
	graphsName = ['FCGraph', 'RingGraph', 'SocialGraph']
	graphsDict = {}
	for graphName in graphsName:
		graphsDict[graphName] = []
	for i in range(0, len(graphsName)):
		for epsilon in epsilonList:
			metrics = ultimatumGame(graphs[i], epsilon = epsilon)
			graphsDict[graphsName[i]] += [metrics, ]

	plt.plot(epsilonList, graphsDict['FCGraph'], label = 'Fully Connected Graph')
	plt.plot(epsilonList, graphsDict['RingGraph'], label = 'Ring Graph')
	plt.plot(epsilonList, graphsDict['SocialGraph'], label = 'Social Graph')
	plt.xlabel('Epsilon Value')
	plt.ylabel('p Value')
	plt.legend()
	plt.title('Epsilon Variation')
	plt.show()
	return graphsDict'''

def main():
	#Create networks
	G1 = createFCGraph()
	G2 = createRingGraph()
	G3 = createFileGraph('edges_social.txt')
	graphs = [G1, G2, G3]

	#Calculate metrics and draw graphics
	graphsDict = getResults(graphs)
	print(graphsDict)
	getGraphs(graphsDict)

if __name__ == '__main__':
	main()