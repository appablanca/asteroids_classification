# 🚀 Asteroid Hazard Classification with Machine Learning

This project applies **machine learning classification techniques** to NASA’s open asteroid observation dataset to predict whether an asteroid is **potentially hazardous to Earth**. The goal is to demonstrate a complete end-to-end data science workflow, including data preprocessing, feature engineering, model training, and evaluation.

This repository is suitable as a **portfolio project** for roles related to:
- Data Science  
- Machine Learning  
- Applied AI  
- Scientific Computing  

---

## 🔍 Project Overview

Near-Earth Objects (NEOs), especially asteroids, can pose potential risks to Earth. NASA provides open datasets containing physical and orbital parameters of asteroids. In this project, we:

- Explore and clean real-world astronomical data  
- Engineer meaningful features  
- Train a classification model  
- Evaluate model performance in predicting hazardous asteroids  

---

## 📁 Repository Structure

```bash
.
├── classification_asteroid.ipynb   # Jupyter notebook with full analysis & modeling
├── nasa.csv                        # NASA asteroid dataset
└── README.md                       # Project documentation
```

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – data manipulation  
- **NumPy** – numerical computing  
- **Scikit-learn** – machine learning  
- **Matplotlib / Seaborn** – visualization  
- **Jupyter Notebook**

---

## ⚙️ How to Run

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

2. Launch the notebook:
```bash
jupyter notebook classification_asteroid.ipynb
```

3. Run the cells to:
   - Load and explore the dataset  
   - Train the classification model  
   - Evaluate results  

---

## 📊 Dataset

The dataset is based on **NASA’s Near-Earth Object (NEO) observations**.  
Example features include:

- `absolute_magnitude_h`  
- `estimated_diameter_min`  
- `estimated_diameter_max`  
- `relative_velocity`  
- `miss_distance`  
- `is_potentially_hazardous_asteroid` (target label)

---

## 🤖 Modeling Approach

The notebook demonstrates:

- Data cleaning & preprocessing  
- Feature selection  
- Train-test split  
- Supervised classification  
- Model evaluation using metrics such as accuracy and confusion matrix  

---

## 📈 Results

The trained model predicts whether an asteroid is **potentially hazardous** based on its physical and orbital characteristics.  
This project illustrates how machine learning can be applied to **real-world scientific datasets** in the context of space and risk analysis.

---

## 💡 Future Improvements

- Try advanced models (Random Forest, Gradient Boosting, XGBoost)  
- Hyperparameter tuning  
- Handle class imbalance (e.g., SMOTE)  
- ROC-AUC and precision-recall analysis  
- Feature importance and model interpretability  

---

## 📌 Why This Project?

This project demonstrates:
- End-to-end machine learning workflow  
- Working with noisy, real-world data  
- Binary classification problem formulation  
- Practical ML applied to a scientific domain (astronomy)

---

## 👤 Author

Developed as a personal machine learning portfolio project.  
Feel free to explore, fork, and improve the repository.
