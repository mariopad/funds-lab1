import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import time
import seaborn as sns

# Parallelizable, but worth it??
def initialize_centroids(data, K):
    indices = np.random.choice(len(data), size=K, replace=False)
    return data[indices].astype(np.float32)

# Xcj = [[x1, y1], ..., [xj,yj]]
# Xi = data = [[x1, y1], ..., [xi,yi]]

# No hacer paralelizable porque len(a)=2 -> pero se puede hacer mas rapida con linalg
#def euc_distance(a, b):
    # Asumo a ~ b
#    foo = 0
#    for i in range(len(a)):
#        foo += (a[i] - b[i])**2
#    return foo ** .5

# Poner dentro de assign_clusters para no perder el array de vainas
#def compute_wcss(wcss_array):
#    wcss = sum(wcss_array)
#    return wcss

#def assign_clusters(data, centroid_array):
#    assignment_array = []
#    wcss_array = []
#    for point in data:
#        distances_array = []
#        for centroid in centroid_array:
#            distances_array.append(euc_distance(point, centroid))
#        selected_cluster = np.argmin(distances_array)
#        assignment_array.append(selected_cluster)
#        wcss_array.append(distances_array[selected_cluster] ** 2)
#        
#    wcss = compute_wcss(wcss_array)
#    return assignment_array, wcss
# El output me dice a que centro corresponde cada punto
# assignment_array = [5, 0, ..., K-1] con len(A)=len(ass_array)

# Midpoint
#def assign_clusters_sequential(data, centroid_array):
#    assignment_array = []
#    wcss_array = []
#    for point in data:
#        # Vectorized distance calculation for one point to all centroids
#        distances = np.linalg.norm(centroid_array - point, axis=1)
#        selected_cluster = np.argmin(distances)
#        assignment_array.append(selected_cluster)
#        wcss_array.append(distances[selected_cluster] ** 2)
#    wcss = sum(wcss_array)
#    return assignment_array, wcss


#def assign_clusters(data, centroid_array):
#    distances = np.linalg.norm(data[:, np.newaxis, :] - centroid_array[np.newaxis, :, :], axis=2)
#    assignment_array = np.argmin(distances, axis=1)
#    min_distances = distances[np.arange(len(data)), assignment_array]
#    wcss = np.sum(min_distances ** 2)
#    return assignment_array, wcss

def assign_clusters(data, centroids):
    # Squared norms of data and centroids
    data_norms = np.sum(data**2, axis=1).reshape(-1, 1)     # shape (N, 1)
    centroid_norms = np.sum(centroids**2, axis=1).reshape(1, -1)  # shape (1, K)

    # Dot product between all data points and centroids
    dot_products = np.dot(data, centroids.T)                # shape (N, K)

    # Squared Euclidean distance matrix
    dists_squared = data_norms + centroid_norms - 2 * dot_products

    # Ensure numerical stability (eliminate tiny negatives due to float precision)
    dists_squared = np.maximum(dists_squared, 0)

    # Assign to closest centroid
    assignment_array = np.argmin(dists_squared, axis=1)

    # Compute WCSS as sum of min distances squared
    min_dists = dists_squared[np.arange(len(data)), assignment_array]
    wcss = np.sum(min_dists)

    return assignment_array, wcss


#def update_centroids(data, K, assignment_array):
#    # Por si acaso aumentamos a una dimensión más, que no esté
#    #hardcodeado
#    n_features = data.shape[1]
#    new_centroids = np.zeros((K, n_features), dtype=mp.float32)
#    for ki in range(K):
#        cluster_points = data[assignment_array == ki]
#
#        if len(cluster_points)>0:
#            new_centroids[ki] = np.mean(cluster_points, axis=0)
#        else:
#            new_centroids[ki] = data[np.random.randint(0, len(data))]
#    return new_centroids

# big for loops are super slow in python
def update_centroids(data, K, assignments):
    n_features = data.shape[1]
    new_centroids = np.zeros((K, n_features), dtype=np.float32)
    counts = np.bincount(assignments, minlength=K)

    # Avoid division by zero
    empty_clusters = (counts == 0)
    counts[empty_clusters] = 1
       
    # Sum each cluster's points. n_features low, better to keep
    for dim in range(n_features):
        new_centroids[:, dim] = np.bincount(assignments, weights=data[:, dim], minlength=K)

    # Divide by count to get mean
    new_centroids /= counts[:, None]

    #Reinitialize empty clusters if any
    if empty_clusters.any():
        new_centroids[empty_clusters] = data[np.random.choice(len(data), size=empty_clusters.sum())]
     
    return new_centroids


#def check_convergence(old_centroids, new_centroids, tol=1e-4):
#    return np.linalg.norm(old_centroids - new_centroids) < tol

# Slightly faster
def check_convergence(old_centroids, new_centroids, tol=1e-4):
    diff = old_centroids - new_centroids
    return np.sum(diff * diff) < tol ** 2

def k_means_sequential(data, K, max_iters=500, tol=1e-4):
    old_centroids = initialize_centroids(data, K)
    for _ in range(max_iters):
        assignment_array, wcss = assign_clusters(data, old_centroids)
        new_centroids = update_centroids(data, K, assignment_array)
        condition_check = check_convergence(old_centroids, new_centroids, tol)
        if condition_check:
            break
        old_centroids = new_centroids
    return assignment_array, old_centroids, wcss


#def distance_to_line(sline_params, point):
#    num = abs(sline_params[0] * point[0] + sline_params[1] * point[1] + sline_params[-1])
#    den = (sline_params[0] ** 2 + sline_params[1] ** 2) ** .5
#    return num / den

def distance_to_line(x0, y0, x1, y1, x2, y2):
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den

def elbow_method(data, K_max, max_iters, tol, sample_size=1000):
    #Por inspección de los datos, podemos ver que el patrón
    # se aprecia bastante bien con tan solo 1e3 datos.
    #Por esto, consideramos que es un overkill usar todo
    # el dataset para determinar K_opt
    if len(data) > sample_size:
        data = data[np.random.choice(len(data), size=sample_size, replace=False)]

    #Curva wcss frente a K
    timei = time.time()
    wcss_array = []
    K_array = range(1, K_max+1)
    for ki in K_array:
        _, _, wcssi = k_means_sequential(data, ki, max_iters, tol)
        wcss_array.append(wcssi)
    wcss_array = np.array(wcss_array)
    timef = time.time()
    print(f"Time spent in K loop: {timef-timei}")

    #Calcular la puta distancia de cada punto y guardar el elemento con menor distancia
    timei = time.time()
    x1, y1 = 1, wcss_array[0]
    x2, y2 = K_max, wcss_array[-1]
    distances = []
    for x0, y0 in zip(K_array, wcss_array):
        d = distance_to_line(x0, y0, x1, y1, x2, y2)
        distances.append(d)
    distances = np.array(distances)
    optimal_K = K_array[np.argmax(distances)]
    timef = time.time()
    print(f"Time spent in calculating distances: {timef-timei}")
    
    #Hacer el grafico y guardarlo
    plt.plot(K_array, wcss_array)
    plt.savefig('elbow_graph')
    plt.close()

    return optimal_K

if __name__ == "__main__":
    #Constants
    K_max = 15
    max_iters = 1000
    tol = 1e-4

    #Register start time
    start_time = time.time()

    #Extraer los datos del csv
    timei = time.time()
    proteins_df = pd.read_csv('proteins.csv')#, nrows=100000)
    proteins = proteins_df[['enzyme', 'hydrofob']].values.astype(np.float32) # seleccionamos dos columnas y -> numpy
    timef = time.time()
    print(f"Time spent in extracting the data: {timef-timei}")

    #Elbow graph and optimal K
    optimal_K = elbow_method(proteins, K_max, max_iters, tol)

    print(f"K optima: {optimal_K}")

    #Cluster data using optimal_K
    assignment_array, centroids, _ = k_means_sequential(proteins, optimal_K, max_iters, tol)

    #Find cluster with highest sequence
    max_cluster = np.bincount(assignment_array).argmax()

    #Graphs
    #Cluster with centroids
    plt.scatter(proteins[:, 0], proteins[:, 1], c=assignment_array, cmap='tab10')
    plt.scatter(centroids[:, 0],centroids[:, 1],marker = '^',c = 'red')
    plt.savefig('hola')

    #Heat map clusters' centroids. Normalization needed
    sns.heatmap(centroids, cmap="viridis")
    plt.savefig('heatmap')
    plt.close()
    #Print stuff of the highest sequence cluster

    end_time = time.time()

    #Print total execution time
    print(f"Total time: {end_time - start_time}")
