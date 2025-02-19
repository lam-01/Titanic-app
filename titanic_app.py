import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
import os

class TitanicAnalyzer:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None  # Sẽ khởi tạo sau khi có dữ liệu
        self.poly = None
        self.feature_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        self.is_fitted = False
    
    def load_and_preprocess(self, data_path):
        """Đọc và tiền xử lý dữ liệu"""
        try:
            # Đọc dữ liệu
            self.data = pd.read_csv(data_path)
            
            # Xử lý missing values
            imputer = SimpleImputer(strategy='mean')
            self.data[self.feature_columns] = imputer.fit_transform(self.data[self.feature_columns])
            
            # Chuyển đổi biến categorical
            self.data['Sex'] = (self.data['Sex'] == 'female').astype(int)
            self.feature_columns.append('Sex')
            
            # One-hot encoding cho Embarked
            embarked_dummies = pd.get_dummies(self.data['Embarked'], prefix='Embarked')
            self.data = pd.concat([self.data, embarked_dummies], axis=1)
            self.feature_columns.extend(embarked_dummies.columns)
            
            return self.data
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return None
    
    def split_data(self, train_size=0.7, valid_size=0.15):
        """Chia dữ liệu thành tập train/valid/test"""
        try:
            test_size = 1 - train_size - valid_size
            
            X = self.data[self.feature_columns]
            y = self.data['Survived']
            
            # Split thành train và temp
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, train_size=train_size, random_state=42)
            
            # Split temp thành valid và test
            valid_ratio = valid_size / (valid_size + test_size)
            X_valid, X_test, y_valid, y_test = train_test_split(
                X_temp, y_temp, train_size=valid_ratio, random_state=42)
            
            # Khởi tạo và fit scaler với training data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_valid_scaled = self.scaler.transform(X_valid)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.is_fitted = True
            
            return (X_train_scaled, X_valid_scaled, X_test_scaled,
                    y_train, y_valid, y_test)
                    
        except Exception as e:
            st.error(f"Error in splitting data: {str(e)}")
            return None
    
    def predict_survival(self, input_data):
        """Dự đoán cho dữ liệu mới"""
        try:
            if not self.is_fitted:
                raise Exception("Model is not fitted yet. Please train the model first.")
            
            # Scale input using fitted scaler
            input_scaled = self.scaler.transform(input_data[self.feature_columns])
            
            # Predict
            if self.poly is not None:
                input_poly = self.poly.transform(input_scaled)
                prediction = self.model.predict(input_poly)[0]
            else:
                prediction = self.model.predict(input_scaled)[0]
            
            # Clip prediction to [0, 1]
            return max(min(prediction, 1), 0)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

def create_streamlit_app():
    st.title("Titanic Survival Analysis")
    
    # Sidebar menu
    st.sidebar.title("Menu")
    menu_option = st.sidebar.radio("Chọn chức năng:", ["Các phương pháp xử lí ", "Dự đoán","Mlflow"])
    
    if menu_option == "Các phương pháp xử lí ":
        st.header("Các phương pháp xử lí titanic.csv")
        st.write("""
        - **Xử lý missing values**: Sử dụng SimpleImputer để điền giá trị trung bình cho các cột số.
        - **Chuyển đổi biến categorical**: Chuyển đổi cột 'Sex' thành binary (0 cho male, 1 cho female).
        - **One-hot encoding**: Áp dụng one-hot encoding cho cột 'Embarked'.
        - **Chuẩn hóa dữ liệu**: Sử dụng StandardScaler để chuẩn hóa dữ liệu.
        """)
    
    elif menu_option == "Dự đoán":
        st.header("Dự đoán khả năng sống sót trên tàu Titanic")
        
        uploaded_file = st.file_uploader("Upload Titanic dataset", type="csv")
        analyzer = TitanicAnalyzer()
        
        if uploaded_file is not None:
            # Load và preprocess data
            data = analyzer.load_and_preprocess(uploaded_file)
            
            if data is not None:
                st.success("Data loaded and preprocessed successfully!")
                
                # Parameters
                st.sidebar.header("Training Parameters")
                train_size = st.sidebar.slider("Training Set Size", 0.5, 0.8, 0.7)
                valid_size = st.sidebar.slider("Validation Set Size", 0.1, 0.25, 0.15)
                degree = st.sidebar.selectbox("Polynomial Degree", [1, 2, 3])
                
                if st.button("Train Model"):
                    # Split data
                    splits = analyzer.split_data(train_size, valid_size)
                    
                    if splits is not None:
                        X_train, X_valid, X_test, y_train, y_valid, y_test = splits
                        
                        with mlflow.start_run():
                            # Train model
                            if degree > 1:
                                analyzer.poly = PolynomialFeatures(degree=degree)
                                X_train_poly = analyzer.poly.fit_transform(X_train)
                                X_valid_poly = analyzer.poly.transform(X_valid)
                                
                                analyzer.model = LinearRegression()
                                analyzer.model.fit(X_train_poly, y_train)
                                y_pred = analyzer.model.predict(X_valid_poly)
                            else:
                                analyzer.model = LinearRegression()
                                analyzer.model.fit(X_train, y_train)
                                y_pred = analyzer.model.predict(X_valid)
                            
                            # Calculate metrics
                            mse = mean_squared_error(y_valid, y_pred)
                            r2 = r2_score(y_valid, y_pred)
                            
                            # Log metrics
                            mlflow.log_metrics({
                                "mse": mse,
                                "r2": r2
                            })
                            
                            # Display results
                            st.subheader("Training Results")
                            col1, col2 = st.columns(2)
                            col1.metric("MSE", f"{mse:.4f}")
                            col2.metric("R2 Score", f"{r2:.4f}")
                            
                            # Plot
                            fig = px.scatter(
                                x=y_valid, y=y_pred,
                                labels={'x': 'Actual', 'y': 'Predicted'},
                                title='Actual vs Predicted Values'
                            )
                            st.plotly_chart(fig)
                
                # Prediction interface
                st.subheader("Prediction Interface")
                col1, col2 = st.columns(2)
                
                with col1:
                    pclass = st.selectbox("Passenger Class", [1, 2, 3])
                    age = st.number_input("Age", 0, 100, 30)
                    sex = st.selectbox("Sex", ["male", "female"])
                
                with col2:
                    sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
                    parch = st.number_input("Parents/Children", 0, 10, 0)
                    fare = st.number_input("Fare", 0.0, 500.0, 32.0)
                    embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
                
                if st.button("Predict"):
                    # Create input DataFrame
                    input_data = pd.DataFrame({
                        'Pclass': [pclass],
                        'Age': [age],
                        'SibSp': [sibsp],
                        'Parch': [parch],
                        'Fare': [fare],
                        'Sex': [1 if sex == 'female' else 0],
                        'Embarked_C': [1 if embarked == 'C' else 0],
                        'Embarked_Q': [1 if embarked == 'Q' else 0],
                        'Embarked_S': [1 if embarked == 'S' else 0]
                    })
                    
                    # Make prediction
                    survival_prob = analyzer.predict_survival(input_data)
                    
                    if survival_prob is not None:
                        st.write(f"Survival Probability: {survival_prob:.2%}")
    elif menu_option=="Mlflow":
        # Tiêu đề
        st.title("Titanic Survival Analysis with MLflow")
        # Hiển thị MLflow Tracking UI trong iframe
        mlflow_url = "http://localhost:5000"  # Thay đổi nếu chạy trên server khác
        st.markdown(f'<iframe src="{mlflow_url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)                    

if __name__ == "__main__":
    create_streamlit_app()
