import numpy as np
from sklearn.decomposition import PCA
import joblib

# 1. load
data = np.load("preprocess_clustering/train_500_embeddings.npy", allow_pickle=True).item()
embeddings = data["embeddings"]  # shape: (500, 1, 768) 일 가능성 있음

# 2. 차원 펴주기
embeddings = np.squeeze(embeddings)  # (500, 768)

# 3. PCA로 차원 축소
target_dim = 20
pca = PCA(n_components=target_dim)
embeddings_pca = pca.fit_transform(embeddings)  # shape: (500, 20)
joblib.dump(pca, "pca_model.pkl")
# 4. 저장할 데이터 구성
save_data = {
    "indices": data["indices"],        # 기존 인덱스 그대로
    "embeddings_pca": embeddings_pca,   # 축소된 임베딩
}

# 5. 저장
np.save("train_500_embeddings_pca.npy", save_data)

print("PCA 변환 완료! 'train_500_embeddings_pca.npy'로 저장했어.")
