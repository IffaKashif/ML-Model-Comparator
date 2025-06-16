# ML-Model-Comparator
An interactive Streamlit web app to compare multiple machine learning models on any uploaded dataset. Whether you're working on a classification or regression problem, this tool provides a simple interface to train, evaluate, and visualize performance metrics across different models.

---

## Features

✅ Automatic task detection (classification vs regression)  
✅ Upload any CSV dataset and select a target variable  
✅ Tune model parameters (KNN, SVM, Random Forest)  
✅ Visual performance comparison via:
- 📈 Model Metrics Table  
- 📊 Confusion Matrix  
- 📉 ROC Curve (for binary classification)

---

## Tech Stack

- **Python**
- **Streamlit**
- **scikit-learn**
- **Plotly**
- **Pandas / NumPy**

---

## 📦 Installation & Usage

1. Clone the repo or download files:
   ```bash
   git clone https://github.com/your-username/ml-model-comparator.git
   cd ml-model-comparator
   
2. Install dependencies:
```bash
pip install -r requirements.txt

3. Run the app:
```bash
streamlit run Ml_Model_Comparator.py
