# ❤️ Heart Disease Prediction using ML Algorithms 🤖  
![Python](https://img.shields.io/badge/Python-3.6+-blue?logo=python) ![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-blue?logo=pandas) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression%20%7C%20Random%20Forest-orange?logo=scikit-learn) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?logo=python) ![Seaborn](https://img.shields.io/badge/Seaborn-Stats%20Plots-blue?logo=python) ![License](https://img.shields.io/badge/License-MIT-yellow) ![Status](https://img.shields.io/badge/Status-Active-brightgreen)

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/3003/3003985.png" alt="Heart Disease Logo" width="140"/>
</p>

🚀 This project builds **machine learning models** to predict the presence of heart disease in patients using clinical features. It compares **Logistic Regression** and **Random Forest** classifiers, performs data preprocessing (handling missing values, feature scaling), and evaluates models using accuracy, confusion matrix, and classification report. The dataset used is the popular [Heart Disease UCI dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) from Kaggle.

---

## ✨ Key Features  
📊 **Data Exploration** – Displays dataset info and first few rows  
⚙️ **Preprocessing** – Handles missing values, feature scaling with `StandardScaler`  
🧠 **Two ML Models** – Logistic Regression and Random Forest  
📈 **Model Evaluation** – Accuracy, confusion matrix, and detailed classification report  
📉 **Comparison** – Identifies the best performing model  
🎨 **Visualization** – Confusion matrix heatmap for the best model  
🔮 **Prediction on New Data** – Example of predicting a single patient record  

---

## 🧠 Tech Stack  
- **Language:** Python 🐍  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Models:** Logistic Regression, Random Forest Classifier  
- **Preprocessing:** StandardScaler  
- **Evaluation:** Accuracy, Confusion Matrix, Classification Report  

---

## 📦 Installation  

```bash
git clone https://github.com/SayabArshad/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
pip install pandas numpy scikit-learn matplotlib seaborn
⚙️ Note: You need to download the heart_Disease.csv dataset (included in the repository) or obtain it from Kaggle.

▶️ Usage
Run the main script:

bash
python "Disease Prediction (e.g., Heart Disease) using ML algorithms.py"
The script will:

Load the dataset.

Display basic info.

Scale features and split data.

Train Logistic Regression and Random Forest models.

Print accuracy, confusion matrix, and classification report for both.

Compare and select the best model.

Show a heatmap of the confusion matrix.

Predict on a new sample patient record.

📁 Project Structure
text
Heart-Disease-Prediction-ML/
│-- Disease Prediction (e.g., Heart Disease) using ML algorithms.py   # Main script
│-- heart_Disease.csv                                                   # Dataset
│-- README.md                                                           # Documentation
│-- assets/                                                             # Images for README
│    ├── code.JPG
│    ├── output.JPG
│    └── plot.JPG
🖼️ Interface Previews
📝 Code Snippet	📊 Console Output
https://assets/code.JPG	https://assets/output.JPG
📈 Confusion Matrix Heatmap
https://assets/plot.JPG
💡 About the Project
Heart disease is one of the leading causes of death worldwide. This project leverages machine learning to assist in early diagnosis by analyzing patient data such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more. The dataset contains 303 samples with 13 features and a binary target (0 = no disease, 1 = disease). Both Logistic Regression and Random Forest models are trained and evaluated. The best model (Logistic Regression in this run) achieves ~85% accuracy. The script also demonstrates how to use the trained model to predict on new, unseen data – a crucial step for real‑world deployment.

🧑‍💻 Author
Developed by: Sayab Arshad Soduzai 👨‍💻
📅 Version: 1.0.0
📜 License: MIT License

⭐ Contributions
Contributions are welcome! Fork the repository, open issues, or submit pull requests to enhance functionality (e.g., hyperparameter tuning, adding more models, feature engineering, or building a web interface).
If you find this project helpful, please ⭐ star the repository to show your support.

📧 Contact
For queries, collaborations, or feedback, reach out at sayabarshad789@gmail.com

❤️ Predicting heart disease with data to save lives.
