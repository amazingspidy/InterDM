import torch
import clip
import numpy as np
from tqdm import tqdm
import os

# 경로 설정
txt_list_path = "data/interhuman_processed/train_500.txt"
text_folder = "data/interhuman_processed/annots"  # 텍스트 파일들이 저장된 폴더
save_path = "train_500_embeddings.npy"

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
token_embedding = clip_model.token_embedding
clip_transformer = clip_model.transformer
positional_embedding = clip_model.positional_embedding
ln_final = clip_model.ln_final
clip_dtype = clip_model.dtype

# 필요한 파트 freeze
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# 인덱스 로드
with open(txt_list_path, "r") as f:
    indices = [line.strip() for line in f if line.strip()]

embeddings = []
index_list = []

for idx in tqdm(indices, desc="Embedding texts"):
    text_file = os.path.join(text_folder, f"{idx}.txt")

    if not os.path.exists(text_file):
        print(f"Warning: {text_file} not found. Skipping.")
        continue

    # 텍스트 읽기
    with open(text_file, "r") as f:
        lines = f.readlines()
        text = " ".join(line.strip() for line in lines if line.strip())

    # CLIP 임베딩 추출
    with torch.no_grad():
        tokenized = clip.tokenize([text], truncate=True).to(device)
        x = token_embedding(tokenized).type(clip_dtype)
        x = x + positional_embedding.type(clip_dtype)
        x = x.permute(1, 0, 2)  # (NLD -> LND)
        x = clip_transformer(x)
        x = x.permute(1, 0, 2)
        clip_out = ln_final(x).type(clip_dtype)
        feature = clip_out[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)]

    embeddings.append(feature.cpu().numpy())
    index_list.append(int(idx))

# numpy로 변환
embeddings = np.stack(embeddings, axis=0)
index_list = np.array(index_list)

# 저장할 형태는 (index, embedding)
result = {
    "indices": index_list,
    "embeddings": embeddings,
}

np.save(save_path, result)
print(f"Saved embeddings to {save_path}")
