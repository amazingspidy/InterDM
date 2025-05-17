import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 1. 데이터 로드
data = np.load("preprocess_clustering/train_500_embeddings_pca.npy", allow_pickle=True).item()
embeddings_pca = data["embeddings_pca"]  # shape: (500, 20)
indices = data["indices"]

# 2. KMeans 클러스터링 (클러스터 수는 필요에 따라 조정)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_pca)

# 3. KNN 학습 (클러스터 라벨을 사용해 분류기로 학습)
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(embeddings_pca, cluster_labels)

# 4. 결과 저장
save_data = {
    "indices": indices,
    "embeddings_pca": embeddings_pca,
    "cluster_labels": cluster_labels,
}
np.save("train_500_embeddings_pca_kmeans.npy", save_data)
print(f"KMeans 클러스터링 완료! 'train_500_embeddings_pca_kmeans.npy'로 저장했어.")

# 5. 인덱스-클러스터 매핑 저장
index_to_cluster = {idx: label for idx, label in zip(indices, cluster_labels)}
np.save("train_500_index_to_cluster_kmeans.npy", index_to_cluster)
print("인덱스와 클러스터 번호를 매핑한 딕셔너리가 저장되었습니다.")

# 6. KMeans 및 KNN 모델 저장 (평가 시 불러오기 위해)
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(knn, "knn_cluster_predictor.pkl")
print("KMeans 모델과 KNN 분류기가 저장되었습니다.")
