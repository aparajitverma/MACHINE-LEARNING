K means clustering

k-means clustering is a method of vector quantization, originally from signal processing, 
that aims to partition n observations into k clusters in which each observation belongs 
to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. 

K Nearest Neighbors - Classification
K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a 
similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already 
in the beginning of 1970’s as a non-parametric technique
Algorithm		
A case is classified by a majority vote of its neighbors, with the case being assigned to the class most common amongst its 
K nearest neighbors measured by a distance function. If K = 1, then the case is simply assigned to the class of its nearest neighbor. 

Data Set Information:
1. Class: no-recurrence-events, recurrence-events
2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
3. menopause: lt40, ge40, premeno.
4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
6. node-caps: yes, no.
7. deg-malig: 1, 2, 3.
8. breast: left, right.
9. breast-quad: left-up, left-low, right-up, right-low, central.
10. irradiat: yes, no.
