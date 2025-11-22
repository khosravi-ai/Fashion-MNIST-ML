# Fashion-MNIST Classification with XGBoost

## 📌 Project Overview
This project applies **XGBoost** to the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), a benchmark dataset consisting of Zalando's article images.  
The dataset contains **28x28 grayscale images** of 10 different clothing categories, making it a drop-in replacement for the classic MNIST dataset of handwritten digits.

The goal of this project is to build a machine learning model that can accurately classify fashion items into their respective categories.

---

## 🗂 Dataset
- **Training set:** 60,000 images  
- **Test set:** 10,000 images  
- **Classes (0–9):**
  - 0 → T-shirt/top  
  - 1 → Trouser  
  - 2 → Pullover  
  - 3 → Dress  
  - 4 → Coat  
  - 5 → Sandal  
  - 6 → Shirt  
  - 7 → Sneaker  
  - 8 → Bag  
  - 9 → Ankle boot  

Each image is flattened into a **784-dimensional vector** (28 × 28 pixels), and the label corresponds to one of the above categories.

---

## ⚙️ Model
The project uses **XGBoost (Extreme Gradient Boosting)** with the following configuration:

```python
model_xgb = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=10,
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=3,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=2,
    random_state=42,
    eval_metric="mlogloss",
    early_stopping_rounds=50
)

Training/Validation split was applied on the training data to enable early stopping.

Final evaluation was done on the separate test set (10,000 samples).

📊 Results

Model performance on the test set:

Accuracy: ~90%

Best classes: Trouser (0.99 F1), Sandal, Bag, Ankle Boot (>0.95 F1)

Challenging classes: Shirt (0.71 F1), Pullover & Coat (~0.83 F1)

Classification Report (summary):

Metric	Score
Accuracy	0.90
Macro Avg	0.90
Weighted Avg	0.90