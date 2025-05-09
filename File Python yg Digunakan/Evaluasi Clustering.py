from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate(labels, X):
    if len(set(labels)) > 1:
        print("Silhouette Score:", silhouette_score(X, labels))
        print("Calinski-Harabasz Index:", calinski_harabasz_score(X, labels))
        print("Davies-Bouldin Index:", davies_bouldin_score(X, labels))
    else:
        print("Tidak bisa dievaluasi (hanya satu cluster ditemukan)")

evaluate(kmeans_labels, X_scaled)
evaluate(dbscan_labels, X_scaled)