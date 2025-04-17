# 🧠 Deep Unlearning: Forgetting the Elephant 🐘

This project demonstrates a deep unlearning approach to selectively forget the **"Elephant"** class from a model trained on a large-scale animal dataset. We use a GAN-style setup, involving a generator and discriminator, to erase the influence of the elephant class while preserving the model’s performance on all other classes.

---

## 📦 Dataset

We use the [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) from Kaggle.

- 🐾 90 animal classes
- 🖼️ 30,000+ images
- 🐘 Target class for forgetting: `elephant`
- 📁 Folder-structured dataset compatible with `torchvision.datasets.ImageFolder`

---

## 🎯 Objective

The goal is to **forget** a specific class ("elephant") from the model’s memory while ensuring the classifier retains performance on the rest of the dataset.

This is done by:
- Training on all classes initially
- Performing **targeted unlearning** on the elephant class
- Evaluating how well the model forgets the target without harming other predictions

---

## ⚙️ Methodology

### 🔨 Data Loading & Preprocessing

- All images are resized to **128×128**
- Transformed to tensors via PyTorch
- Dataset split:
  - `retain_loader`: All classes **except elephant**
  - `forget_loader`: Only **elephant** images

### 🧠 Model Components

| Component        | Description                                      |
|------------------|--------------------------------------------------|
| `generator`      | ResNet18 model learning to mask elephant features |
| `discriminator`  | Binary classifier separating retained/forgotten  |
| `feature_extractor` | ResNet18 used to extract consistent features  |

---

### 🌀 Unlearning Framework (GAN-like)

- **Discriminator** tries to classify whether features are from "retained" or "forgotten" class
- **Generator** is optimized to make elephant features look like non-elephant ones
