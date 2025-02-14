import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Titanic App 🤖")
# Tiền xử lý dữ liệu
with st.expander("Data Preprocessing") : 

# Đọc dữ liệu
    df = pd.read_csv("titanic.csv")
st.write(f"**Kích thước dữ liệu :** {df.shape}")

# Kiểm tra giá trị thiếu
    missing_values_before = df.isnull().sum()
    st.write("**Giá trị thiếu trước xử lý:**")
    st.write(missing_values_before.to_frame().T)

# Xử lý dữ liệu thiếu
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Cabin"].fillna("Unknown", inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
# Kiểm tra giá trị thiếu sau xử lý
    missing_values_after = df.isnull().sum()
    st.write("**Giá trị thiếu sau xử lý:**")
    st.write(missing_values_after.to_frame().T)

# Kiểm tra và xử lý dữ liệu trùng lặp
    duplicates_before = df.duplicated().sum()
    st.write(f"**Số bản ghi trùng trước khi xử lý:** {duplicates_before}")
    df.drop_duplicates(inplace=True)
    duplicates_after = df.duplicated().sum()
    st.write(f"**Số bản ghi trùng sau khi xử lý:** {duplicates_after}")

# Chuyển đổi biến phân loại thành số
    label_enc = LabelEncoder()
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = label_enc.fit_transform(df[col])

    st.write("**Dữ liệu sau khi xử lý:**")
    st.write(df.head())

with st.expander("Data Visualization") :
# Đọc dữ liệu kích thước tập dữ liệu
    split_info = {
        "Train": len(pd.read_csv("train.csv")),
        "Validation": len(pd.read_csv("valid.csv")),
        "Test": len(pd.read_csv("test.csv"))
    }

    # Chuyển đổi thành DataFrame để trực quan hóa
    df_split = pd.DataFrame(list(split_info.items()), columns=["Dataset", "Size"])

    # Giao diện Streamlit
    st.write("**Số lượng mẫu trong mỗi bộ dữ liệu:**")
    st.table(df_split)

    # Vẽ biểu đồ
    fig = px.bar(df_split, x="Dataset", y="Size", title="Tổng quan về dữ liệu ", color="Dataset")
    st.plotly_chart(fig)

    # Hiển thị ma trận tương quan
    df_train = pd.read_csv("train.csv")
    corr_matrix = df_train.corr(numeric_only=True)
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Ma trận tương quan ")
    st.plotly_chart(fig_corr)
    # Hiển thị kết quả huấn luyện mô hình
    with open("model_results.txt", "r") as f:
        model_results = f.read()
    st.write("**Model Training Results:**")
    st.text(model_results)

