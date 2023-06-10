import networkx as nx
from node2vec import Node2Vec

# Create a graph
graph = nx.Graph()
node_num = 5298
with open('data/adjlist.csv', 'r') as f:
    graph.add_nodes_from(range(node_num))
    count = 0
    for line in f.readlines():
        neighbor_list = line.strip().split(',')
        node_id = int(neighbor_list[0])
        for j in range(1, len(neighbor_list)):
            if neighbor_list[j] != "":
                neighbor_id = int(neighbor_list[j])
                graph.add_edge(node_id, int(neighbor_id))
            else:
                count += 1
    print(count)
print("Graph created")
print(nx.info(graph))

# Precompute probabilities and generate walks
n2v = Node2Vec(graph, dimensions=256, walk_length=30, num_walks=200, workers=8)

# Embed nodes
model = n2v.fit(window=10, min_count=1, batch_words=4, workers=8)

# Save embeddings for later use
model.wv.save_word2vec_format("data/node2vec_256.emb")

print(model.wv.vectors.shape)
