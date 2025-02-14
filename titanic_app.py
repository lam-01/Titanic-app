# app.py - Streamlit visualization
import streamlit as st
import pandas as pd
import plotly.express as px

# Đọc dữ liệu kích thước tập dữ liệu
split_info = {
    "Train": len(pd.read_csv("train.csv")),
    "Validation": len(pd.read_csv("valid.csv")),
    "Test": len(pd.read_csv("test.csv"))
}

# Chuyển đổi thành DataFrame để trực quan hóa
df_split = pd.DataFrame(list(split_info.items()), columns=["Dataset", "Size"])

# Giao diện Streamlit
st.title("Data Visualization")
st.write("**Số lượng mẫu trong mỗi bộ dữ liệu:**")
st.table(df_split)

# Vẽ biểu đồ
fig = px.bar(df_split, x="Dataset", y="Size", title="Tổng quan về dữ liệu ", color="Dataset")
st.plotly_chart(fig)

# Hiển thị ma trận tương quan
df_train = pd.read_csv("G:/ML/MLFlow/train.csv")
corr_matrix = df_train.corr(numeric_only=True)
fig_corr = px.imshow(corr_matrix, text_auto=True, title="Ma trận tương quan ")
st.plotly_chart(fig_corr)
# Hiển thị kết quả huấn luyện mô hình
with open("G:/ML/MLFlow/model_results.txt", "r") as f:
    model_results = f.read()
st.write("**Model Training Results:**")
st.text(model_results)

