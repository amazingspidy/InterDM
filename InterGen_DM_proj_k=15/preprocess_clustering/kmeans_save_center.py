import numpy as np
from sklearn.cluster import KMeans

# ===== 1. PCA 임베딩 데이터 로드 =====
data = np.load("preprocess_clustering/train_500_embeddings_pca.npy", allow_pickle=True).item()
embeddings_pca = data["embeddings_pca"]  # shape: (500, 20)
indices = data["indices"]                # shape: (500,)

# ===== 2. KMeans 클러스터링 수행 =====
num_clusters = 15
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_pca)  # shape: (500,)
cluster_centers = kmeans.cluster_centers_            # shape: (10, 20)

# # ===== 3. 결과 저장 (클러스터 라벨 포함) =====
# clustered_data = {
#     "indices": indices,
#     "embeddings_pca": embeddings_pca,
#     "cluster_labels": cluster_labels,
#     "cluster_centers": cluster_centers,
# }
# np.save("train_500_embeddings_pca_kmeans.npy", clustered_data)
# print("KMeans 클러스터링 완료: 'train_500_embeddings_pca_kmeans.npy' 저장됨")

# # ===== 4. (인덱스 → 클러스터번호) 딕셔너리 생성 및 저장 =====
# index_to_cluster = {int(idx): int(label) for idx, label in zip(indices, cluster_labels)}
# np.save("train_500_index_to_cluster.npy", index_to_cluster)
# print("인덱스별 클러스터 번호 매핑 완료: 'train_500_index_to_cluster.npy' 저장됨")

# ===== (선택) 5. 클러스터 중심 저장 (inference 시 유용) =====
np.save("preprocess_clustering/kmeans_cluster_centers_n=15.npy", cluster_centers)
print("클러스터 중심 저장: 'kmeans_cluster_centers_n=15.npy'")
