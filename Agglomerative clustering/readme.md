The agglomerative clustering is the most common type of hierarchical clustering used to group objects in clusters based on their similarity. It’s also known as AGNES (Agglomerative Nesting). The algorithm starts by treating each object as a singleton cluster.
Next, pairs of clusters are successively merged until all clusters have been merged into one big cluster containing all objects. Th
e result is a tree-based representation of the objects, named dendrogram.
Agglomerative clustering works in a “bottom-up” manner. That is, each object is initially considered as a single-element cluster (leaf).
At each step of the algorithm, the two clusters that are the most similar are combined into a new bigger cluster (nodes). 
