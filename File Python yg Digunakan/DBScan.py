import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dan preprocessing data
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df.drop(columns=["ID", "ZIP Code"], inplace=True)
df = df.dropna()
X = df.drop(columns=["Personal Loan"])
X_scaled = StandardScaler().fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Visualisasi PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=15)
plt.title("DBSCAN Clustering (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("dbscan_visualisasi.png")
plt.show()
