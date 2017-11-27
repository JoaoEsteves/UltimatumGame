import networkx as nx
from random import uniform, randint

def createGraph(file):
    G = nx.read_edgelist(file, create_using = nx.Graph(), nodetype = int)
    G.add_nodes_from(G.nodes, r = 0)

    for node in G.nodes:
        G.add_nodes_from([node], p = round(uniform(0,1), 2))
        G.add_nodes_from([node], q = round(uniform(0,1), 2))

    return G

def meanMetrics(G):

    qMean = 0
    pMean = 0
    rMean = 0
    for node in G.nodes:
        pMean += G.node[node]['p']
        qMean += G.node[node]['q']
        rMean += G.node[node]['r']

    return pMean/len(G.nodes), qMean/len(G.nodes), rMean/len(G.nodes)

def main():

    G = createGraph('fc.txt')
    metrics = meanMetrics(G)
    print('before', metrics)

    for n in range(0, 50):
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                if G.node[node]['p'] >= G.node[neighbor]['q']:
                    G.node[node]['r'] += G.node[neighbor]['q']
                    G.node[neighbor]['r'] += (1 - G.node[neighbor]['q'])

        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            chosenNeighbor = neighbors[randint(0, len(neighbors) - 1)]

            if G.node[node]['r'] < G.node[chosenNeighbor]['r']:
                G.node[node]['p'] = G.node[chosenNeighbor]['p']
                G.node[node]['q'] = G.node[chosenNeighbor]['q']

    metrics = meanMetrics(G)
    print('after', metrics)



if __name__ == "__main__":
    main()
