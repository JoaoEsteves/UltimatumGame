import networkx as nx
import numpy as np
from random import uniform, randint
import signal

def signal_handler(signal, frame):
	global interrupted
	interrupted = True

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

def setGraphValues(G):
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
	return (round(pMean/len(G.nodes), 2),round(qMean/len(G.nodes), 2),round(rMean/len(G.nodes), 2))

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

def play(G):
	#print(meanMetrics(G))
	iterations = 100
	pMeanPerNode = []
	#Valores medios de p por iteracao
	evolution = []
	#valores iniciais da media de p'''
	evolution.append(meanMetrics(G)[0])
	for node in G.nodes:
		pMeanPerNode[node] = G.node[node]['p']

	for n in range(0, iterations):
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
				noiseNode = noiseUG2()
				if G.node[chosenNeighbor]['p'] + noiseNode > 0:
					newP = G.node[chosenNeighbor]['p'] + noiseNode
				else:
					newP = 0
				noiseNeighbor = noiseUG2()
				if G.node[chosenNeighbor]['q'] + noiseNeighbor > 0:
					newQ = G.node[chosenNeighbor]['q'] + noiseNeighbor
				else:
					newQ = 0
				updatedDict[node] = (newP, newQ)

		for node in G.nodes:
			pMeanPerNode[node] += G.node[node]['p']

		updateMetrics(G, updatedDict)
		#soma de valores medios de p por iteracao'''
		evolution.append[meanMetrics(G)[0]]
		#print(meanMetrics(G))
		resetReward(G)

	for i in pMeanPerNode:
		pMeanPerNode[i] = pMeanPerNode[i] / iterations

	return meanMetrics(G), evolution, pMeanPerNode

#def main():	 nao consigo fazer o signal com a main, nao me apetece pensar no assunto
run = True
signal.signal(signal.SIGINT, signal_handler)
interrupted = False

while(run):
	#Create Graphs
	G1 = createFCGraph()
	G2 = createRingGraph()
	G3 = createSocialGraph()
	#G4 = createGridGraph()
	setGraphValues(G1)
	setGraphValues(G2)
	setGraphValues(G3)
	#setGraphValues(G4)


	#run simulations
	result = play(G1)
	print(result[0], '------' ,result[1, '------', result[2]])


	if interrupted:
		run = False


#if __name__ == '__main__':
#	main()




'''
def main():
	#Create Graphs
	G = createFileGraph('edges_fc.txt')
	setGraphValues(G)
	#print(G.nodes(data=True))
	#G = createFCGraph()
	#G = createRingGraph()
	#G = createSocialGraph()


	pMean = 0

	#File Graph
	for n in range(0,50):
		pMean += play(G)[0]
	print('<p>: ' + pMean/50)

	#FC Graph

	#Ring Graph

	#Social Graph
'''
