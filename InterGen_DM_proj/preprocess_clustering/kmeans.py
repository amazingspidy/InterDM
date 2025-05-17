# import numpy as np
# from sklearn.cluster import KMeans

# # 1. load
# data = np.load("preprocess_clustering/train_500_embeddings_pca.npy", allow_pickle=True).item()

# embeddings_pca = data["embeddings_pca"]  # shape: (500, 20)


# indices = data["indices"]

# # 2. KMeans 클러스터링
# num_clusters = 5
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(embeddings_pca)  # shape: (500,)

# # 3. 결과 저장
# save_data = {
#     "indices": indices,
#     "embeddings_pca": embeddings_pca,
#     "cluster_labels": cluster_labels,  # 클러스터 결과 추가
# }

# np.save("train_500_embeddings_pca_kmeans_n=5.npy", save_data)

# print(f"KMeans로 10개 클러스터링 완료! 'train_500_embeddings_pca_clustered.npy'로 저장했어.")


import numpy as np

# KMeans 결과에서 가져온 데이터 (예시)
data = np.load("preprocess_clustering/train_500_embeddings_pca_kmeans_n=5.npy", allow_pickle=True).item()
indices = data["indices"]  # 데이터 인덱스
cluster_labels = data["cluster_labels"]  # KMeans 클러스터 번호

# 2. (index, cluster_number) 형태로 딕셔너리 생성
index_to_cluster = {idx: label for idx, label in zip(indices, cluster_labels)}

# 3. 딕셔너리 저장
np.save("train_500_index_to_cluster_n=5.npy", index_to_cluster)

print("인덱스와 클러스터 번호를 매핑한 딕셔너리가 저장되었습니다.")

