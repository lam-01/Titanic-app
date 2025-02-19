import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

class TitanicAnalyzer:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None
        self.poly = None
        self.feature_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        self.is_fitted = False
    
    def load_and_preprocess(self, data):
        imputer = SimpleImputer(strategy='mean')
        self.data = data.copy()
        self.data[self.feature_columns] = imputer.fit_transform(self.data[self.feature_columns])
        self.data['Sex'] = (self.data['Sex'] == 'female').astype(int)
        self.feature_columns.append('Sex')
        embarked_dummies = pd.get_dummies(self.data['Embarked'], prefix='Embarked')
        self.data = pd.concat([self.data, embarked_dummies], axis=1)
        self.feature_columns.extend(embarked_dummies.columns)
        return self.data
    
    def split_data(self, train_size=0.7, valid_size=0.15):
        test_size = 1 - train_size - valid_size
        X = self.data[self.feature_columns]
        y = self.data['Survived']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
        valid_ratio = valid_size / (valid_size + test_size)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, train_size=valid_ratio, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        return X_train_scaled, X_valid_scaled, y_train, y_valid
    
    def train_model(self, X_train, y_train, degree=1):
        if degree > 1:
            self.poly = PolynomialFeatures(degree=degree)
            X_train = self.poly.fit_transform(X_train)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
    
    def predict_survival(self, input_data):
        if not self.is_fitted:
            return "Model chưa được huấn luyện!"
        input_scaled = self.scaler.transform(input_data[self.feature_columns])
        prediction = self.model.predict(input_scaled)[0]
        return round(prediction)

def create_streamlit_app():
    st.title("Titanic Survival Prediction")
    menu = st.sidebar.radio("Menu", ["Phương pháp & Kết quả tiền xử lý", "Dự đoán"])
    
    default_data_path = "G:/ML/MLFlow/my_env/titanic.csv"
    data = pd.read_csv(default_data_path)
    analyzer = TitanicAnalyzer()
    processed_data = analyzer.load_and_preprocess(data)
    
    if menu == "Phương pháp & Kết quả tiền xử lý":
        st.header("Kết quả tiền xử lý")
        st.subheader("Dữ liệu gốc")
        st.write(data.head())
        st.subheader("Dữ liệu sau khi tiền xử lý")
        st.write(processed_data.head())
    
    elif menu == "Dự đoán":
        st.header("Dự đoán sống sót")
        train_size = st.sidebar.slider("Train Size", 0.5, 0.8, 0.7)
        valid_size = st.sidebar.slider("Validation Size", 0.1, 0.3, 0.15)
        degree = st.sidebar.selectbox("Polynomial Degree", [1, 2, 3])
        X_train, X_valid, y_train, y_valid = analyzer.split_data(train_size, valid_size)
        analyzer.train_model(X_train, y_train, degree)
        y_pred = analyzer.model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)
        st.metric("MSE", f"{mse:.4f}")
        st.metric("R2 Score", f"{r2:.4f}")
    
if __name__ == "__main__":
    create_streamlit_app()
