# 🤖 InterGen++  
**Enhancing Human Motion Generation via Clustering-Guided Training**

---

## 🎯 Project Goals

> Leverage clustering techniques to enhance the training and inference stages of human motion generation models.

- 📌 Cluster human motion data (e.g., using K-Means) to inject structural priors  
- 📌 Improve semantic alignment and motion diversity via cluster-based guidance  
- 📌 Propose and evaluate multiple methods for integrating cluster cues into a diffusion-based generation pipeline

---

## 🗂️ Dataset

📦 **InterHuman Dataset**  
[📥 Download Link (Google Drive)](https://drive.google.com/drive/folders/1oyozJ4E7Sqgsr7Q747Na35tWo5CjNYk3)

---

## 📁 Checkpoints

1. [📥 Download Pretrained Checkpoint](https://drive.google.com/drive/folders/1ojxlLLud2dJaMmTBovWE6-2SPRX7FQmD)  
2. Create a `checkpoints/` folder in the root directory  
3. Place the downloaded file inside `/checkpoints`

---

## ⚙️ Environment Setup

> 🔗 Refer to the official InterGen repository for dependencies and setup:
> [https://github.com/tr3e/InterGen](https://github.com/tr3e/InterGen)

---

## 🚀 How to Run

```bash
# Training
python tools/train.py
```
```bash
# Inference
python tools/infer.py
```
```bash
# Evaluation
python tools/eval.py
```

## 📊 Results & Examples

<table>
  <tr>
    <td align="center"><b>Boxing</b></td>
    <td align="center"><b>Latin Dance</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0867c5e6-87e9-4b74-b158-8ea11bb164a7" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/8836d92b-1599-4e17-9853-4c15a30fd6ce" width="300"/></td>
  </tr>
  <tr>
    <td align="center"><b>Bowing</b></td>
    <td align="center"><b>Embrace</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c21c7b1b-a209-4f9a-8b9a-8d5b35f1b7b4" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/7f2d5908-b517-412e-b659-c790d09f0d51" width="300"/></td>
  </tr>
</table>


