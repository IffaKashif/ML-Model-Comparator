# ML Model Comparator App
# Author: Iffa Kashif
# Description: Streamlit app to compare machine learning models for both classification and regression tasks using metrics and visualizations.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.set_page_config(page_title="ML Model Comparator", layout="wide")

# Page Header
st.markdown("""
    <style>
        .main-title {
            font-size: 35px;
            font-weight: bold;
            color: #4CAF50;
        }
        .section-title {
            font-size: 22px;
            color: #1F4E79;
            margin-top: 20px;
        }
    </style>
    <div class="main-title">🤖 ML Model Comparator</div>
""", unsafe_allow_html=True)

st.markdown("Upload your dataset, tune models, and compare multiple models with performance metrics and visualizations.")

# File Upload
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("<div class='section-title'>📄 Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    target = st.selectbox("🎯 Select Target Column", df.columns)
    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Preprocessing
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        task_type = 'classification' if len(np.unique(y)) <= 10 and y.dtype in ['int', 'int32', 'int64'] else 'regression'

        st.markdown("<div class='section-title'>⚙️ Model Evaluation</div>", unsafe_allow_html=True)
        st.markdown("### 🔧 Choose Model Parameters")

        # expander for parameters
        with st.expander("🎛️ Show/Hide Model Parameters", expanded=False):
          st.subheader("🔹 K-Nearest Neighbors (KNN)")
          knn_k = st.slider("Number of Neighbors", min_value=1, max_value=15, value=5)

          st.subheader("🔹 Random Forest")
          rf_n = st.slider("Number of Trees (Estimators)", min_value=10, max_value=200, value=100, step=10)

          st.subheader("🔹 Support Vector Machine (SVM)")
          svm_c = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)


        if task_type == 'classification':
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=rf_n),
                "KNN": KNeighborsClassifier(n_neighbors=knn_k),
                "SVM": SVC(probability=True, C=svm_c)
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=rf_n),
                "KNN": KNeighborsRegressor(n_neighbors=knn_k),
                "SVM": SVR(C=svm_c)
            }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == 'classification':
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                average_type = 'binary' if len(np.unique(y)) == 2 else 'weighted'
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average=average_type, zero_division=0)
                rec = recall_score(y_test, y_pred, average=average_type, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=average_type)
                roc = roc_auc_score(y_test, y_prob) if y_prob is not None and len(np.unique(y)) == 2 else None

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1,
                    "ROC AUC": roc
                })
            else:
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results.append({
                    "Model": name,
                    "MSE": mse,
                    "MAE": mae,
                    "R2 Score": r2
                })

        results_df = pd.DataFrame(results).set_index("Model")
        st.markdown("<div class='section-title'>📈 Model Performance</div>", unsafe_allow_html=True)
        st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

        if task_type == 'classification':
            st.markdown("<div class='section-title'>📊 Confusion Matrices</div>", unsafe_allow_html=True)
            for name, model in models.items():
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                labels = [str(i) for i in range(cm.shape[0])]
                z_text = [[str(val) for val in row] for row in cm]
                fig = ff.create_annotated_heatmap(cm, x=labels, y=labels, annotation_text=z_text, colorscale='Blues')
                fig.update_layout(title_text=f"{name} - Confusion Matrix", title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)

            if len(np.unique(y)) == 2:
                st.markdown("<div class='section-title'>📉 ROC Curves</div>", unsafe_allow_html=True)
                fig = go.Figure()
                for name, model in models.items():
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
                fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Model comparison complete! Adjust parameters and re-run for tuning.")

else:
    st.info("☝️ Please upload a CSV file to begin.")
