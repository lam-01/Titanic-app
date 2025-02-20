import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
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
        """Đọc và tiền xử lý dữ liệu với MLflow"""
        try:
            mlflow.start_run()
            st.header("**📚Tiền xử lý dữ liệu**")
            
            # Đọc dữ liệu
            st.write("**1. Đọc dữ liệu**")
            self.data = pd.read_csv(data_path)
            mlflow.log_param("initial_data_shape", self.data.shape)
            st.write("Dữ liệu ban đầu:", self.data.head())
            
            # Xử lý missing values
            st.write("**2. Xử lý giá trị bị thiếu**")
            st.write("\n- Điền giá trị thiếu trong 'Age' bằng giá trị trung bình.\n- Điền giá trị thiếu trong 'Embarked' bằng giá trị xuất hiện nhiều nhất.")
            missing_values_before = self.data.isnull().sum().sum()
            st.write("Số lượng dữ liệu bị thiếu : ")
            st.write(self.data.isnull().sum().T)
            imputer = SimpleImputer(strategy='mean')
            self.data[self.feature_columns] = imputer.fit_transform(self.data[self.feature_columns])
            missing_values_after = self.data.isnull().sum().sum()
            mlflow.log_metric("missing_values_before", missing_values_before)
            mlflow.log_metric("missing_values_after", missing_values_after)
            st.write("Dữ liệu sau khi xử lý :", self.data.head())

            # Xóa các cột không cần thiết
            st.write("**3. Xóa các cột không cần thiết: Cabin, Name, Ticket**")
            st.write("\n- Xóa cột 'Name' (Tên hành khách không ảnh hướng trực tiếp đến sống sót hay không).\n- Xóa cột 'Ticket' (Số vé là một chuỗi ký tự không mang nhiều ý nghĩa rõ ràng đối với mô hình dự đoán).\n- Xóa cột 'Cabin' (Dữ liệu bị thiếu quá nhiều ,rất nhiều hành khách không có thông tin về cabin).")
            self.data.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True, errors='ignore')
            mlflow.log_param("after_column_removal_shape", self.data.shape)
            st.write("Dữ liệu sau khi xóa cột:", self.data.head())
            
            # Chuyển đổi biến categorical
            st.write("**4. Chuyển đổi biến phân loại**")
            self.data['Sex'] = (self.data['Sex'] == 'female').astype(int)
            self.feature_columns.append('Sex')
            st.write("Dữ liệu sau khi chuyển đổi 'Sex': male:0 - female: 1 ", self.data[['Sex']].head())
            
            # One-hot encoding cho Embarked
            st.write("**5. One-hot encoding cho Embarked**")
            st.write("""One-Hot Encoding (Mã hóa One-Hot) là một phương pháp chuyển đổi dữ liệu dạng phân loại (categorical data) thành dạng số để sử dụng trong các mô hình học máy.
                Thay vì gán nhãn bằng số nguyên (ví dụ: 0, 1, 2), One-Hot Encoding sẽ tạo ra các cột nhị phân (0 hoặc 1) cho từng giá trị duy nhất trong cột phân loại.""")

            st.image("d1.png")
            st.write("Thay vì gán số nguyên (0,1,2) cho Embarked, ta tạo ra 3 cột mới: Embarked_C, Embarked_Q, Embarked_S")
            embarked_dummies = pd.get_dummies(self.data['Embarked'], prefix='Embarked')
            self.data = pd.concat([self.data, embarked_dummies], axis=1)
            self.feature_columns.extend(embarked_dummies.columns)
            mlflow.log_param("one_hot_encoded_columns", list(embarked_dummies.columns))
            st.write("Dữ liệu sau khi One-Hot Encoding cho 'Embarked':")
            st.write(self.data[embarked_dummies.columns].head())

            
            mlflow.end_run()
            return self.data
            
        except Exception as e:
            st.error(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
            mlflow.end_run(status='FAILED')
            return None

    def split_data(self, train_size=0.7, valid_size=0.15):
        """Chia dữ liệu thành tập train/valid/test và chuẩn hóa"""
        try:
            with mlflow.start_run(run_name="Data_Splitting"):
                
                st.header("**📚Chia dữ liệu thành tập train/valid/test**")
                
                test_size = 1 - train_size - valid_size
                st.write(f"🔸 Tỉ lệ: Train = {train_size}, Valid = {valid_size}, Test = {test_size}")
                
                X = self.data[self.feature_columns]
                y = self.data['Survived']
                
                # Split thành train và temp
                st.write("**1. Chia dữ liệu thành tập train và temp**")
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
                st.write(f"🔹 Kích thước tập train: {X_train.shape}")
                
                # Split temp thành valid và test
                valid_ratio = valid_size / (valid_size + test_size)
                st.write("**2. Chia tập temp thành tập validation và test**")
                X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, train_size=valid_ratio, random_state=42)
                st.write(f"🔸 Kích thước tập valid: {X_valid.shape}")
                st.write(f"🔸 Kích thước tập test: {X_test.shape}")
                
                # Khởi tạo và fit scaler với training data
                st.write("**3. Chuẩn hóa dữ liệu bằng StandardScaler**")
                st.write("StandardScaler chuẩn hóa dữ liệu bằng cách đưa giá trị trung bình về 0 và độ lệch chuẩn về 1.")
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_valid_scaled = self.scaler.transform(X_valid)
                X_test_scaled = self.scaler.transform(X_test)
                
                st.write("📊 **Dữ liệu sau khi chuẩn hóa (5 dòng đầu tiên):**")
                st.write(pd.DataFrame(X_train_scaled[:5], columns=self.feature_columns))
                
                self.is_fitted = True
                
                # Log thông tin vào MLflow
                mlflow.log_param("train_size", X_train.shape)
                mlflow.log_param("valid_size", X_valid.shape)
                mlflow.log_param("test_size", X_test.shape)
                mlflow.log_param("scaling_method", "StandardScaler")
                mlflow.end_run()
                
                return (X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test)
                        
        except Exception as e:
            mlflow.log_param("status", "FAILED")
            st.error(f"❌ Lỗi khi chia dữ liệu: {str(e)}")
            mlflow.end_run(status='FAILED')
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
            
            return prediction
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

def create_streamlit_app():
    st.title("Titanic 🚢")
    
    # Sử dụng st.tabs để tạo thanh menu
    tab2, tab3 = st.tabs([ "🔍 Huấn luyện và Dự đoán", "🚀 MLflow"])
    analyzer = TitanicAnalyzer()
        
    with tab2:

            data_path = "titanic.csv"  # Đường dẫn cố định
            analyzer = TitanicAnalyzer()
            data = analyzer.load_and_preprocess(data_path)
                    
            # Parameters
            st.sidebar.header("Training Parameters")
            train_size = st.sidebar.slider("Training Set Size", 0.5, 0.8, 0.7)
            valid_size = st.sidebar.slider("Validation Set Size", 0.1, 0.25, 0.15)
            degree = st.sidebar.selectbox("Polynomial Degree", [1, 2, 3])
            
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
                        X_test_poly = analyzer.poly.transform(X_test)
                        
                        analyzer.model = LogisticRegression()
                        analyzer.model.fit(X_train_poly, y_train)
                        # Dự đoán
                        y_pred_train = analyzer.model.predict(X_train_poly)
                        y_pred_valid = analyzer.model.predict(X_valid_poly)
                        y_pred_test = analyzer.model.predict(X_test_poly)
                    else:
                        analyzer.model = LogisticRegression()
                        analyzer.model.fit(X_train, y_train)
                    
                    # Luwu mô hình vào session_state
                        st.session_state['model'] = analyzer.model
                        st.session_state['scaler'] = analyzer.scaler
                        st.session_state['poly'] = analyzer.poly
                    # Dự đoán
                        y_pred_train = analyzer.model.predict(X_train)
                        y_pred_valid = analyzer.model.predict(X_valid)
                        y_pred_test = analyzer.model.predict(X_test)
                    
                    # Calculate metrics
                    
                    mse_train = mean_squared_error(y_train, y_pred_train)
                    r2_train = r2_score(y_train, y_pred_train)
                    mse_valid = mean_squared_error(y_valid, y_pred_valid)
                    r2_valid = r2_score(y_valid, y_pred_valid)
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    r2_test = r2_score(y_test, y_pred_test)
                    
                    # Cross-validation
                    y_pred_cv = cross_val_predict(analyzer.model, X_train, y_train, cv=5)
                    mse_cv=mean_squared_error(y_train,y_pred_cv)
                    # r2_cv = r2_score(y_train, y_pred_cv)

                    
                    # Log metrics
                    mlflow.log_metrics({
                        "train_mse": mse_train,
                        "valid_mse": mse_valid,
                        "test_mse": mse_test,
                        # "train_r2": r2_train,
                        # "valid_r2": r2_valid,
                        # "test_r2": r2_test,
                        # "cv_r2": r2_cv,
                        "cv_mse":mse_cv
                    })
                    
                    # Display results
                    st.header("**📊 Huấn luyện mô hình**")
                    st.image("d3.jpg")
                    st.write("**. Đánh giá mô hình**")
                    st.write("- MSE (Mean Squared Error) là Sai số bình phương trung bình là một chỉ số đánh giá hiệu suất của mô hình hồi quy, đo lường mức độ sai lệch giữa giá trị thực tế và giá trị dự đoán.")
                    st.write("Công thức của MSE :")
                    st.image("d4.jpg")
                    st.write("- Cross Validation (CV) là một kỹ thuật trong Machine Learning giúp đánh giá mô hình một cách chính xác bằng cách chia dữ liệu thành nhiều phần để huấn luyện và kiểm tra nhiều lần, thay vì chỉ dùng một tập dữ liệu cố định.")
                    st.image("d2.jpg",width=400)
                    results_df = pd.DataFrame({
                        "Metric": ["MSE (Train)", "MSE (Validation)", "MSE (Test)", "MSE (Cross-Validation)"],
                        "Value": [mse_train, mse_valid, mse_test, mse_cv]
                    })

                    # Hiển thị bảng trong Streamlit
                    st.write("**Kết quả đánh giá mô hình**")
                    st.table(results_df)
                    
        
        # Prediction interface
            st.subheader("Giao diện dự đoán")
            # Kiểm tra nếu mô hình đã huấn luyện trước khi dự đoán
            if 'model' in st.session_state:
                analyzer.model = st.session_state['model']
                analyzer.scaler = st.session_state['scaler']
                analyzer.poly = st.session_state['poly']
                analyzer.is_fitted = True
            else:
                st.error("Vui lòng huấn luyện mô hình trước khi dự đoán!")

            if analyzer.is_fitted:
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
                survival_prediction = analyzer.predict_survival(input_data)
                
                if survival_prediction is not None:
                    st.success(f"Dự đoán : {'Survived' if survival_prediction == 1 else 'Not Survived'}")

    with tab3:
        # Hiển thị MLflow Tracking UI trong iframe
        mlflow_url = "http://localhost:5000"  # Thay đổi nếu chạy trên server khác
        st.markdown(f'<iframe src="{mlflow_url}" width="800" height="400"></iframe>', unsafe_allow_html=True)

if __name__ == "__main__":
    create_streamlit_app()
