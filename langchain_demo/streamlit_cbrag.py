# streamlit_cbrag.py

import streamlit as st
import datetime
from chatbot import get_answer  # Import hàm get_answer từ file chatbot.py
from dotenv import load_dotenv  # Thêm import để sử dụng dotenv
import os  # Thêm import để sử dụng os
import requests  # Thêm import để sử dụng requests


# Tải các biến môi trường từ file .env
load_dotenv()

# Lấy key từ biến môi trường
# api_key = os.getenv("API_KEY")  # Thay thế bằng biến môi trường

# Tiêu đề ứng dụng
st.title("Chatbot Tư Vấn Tuyển Sinh")

# Tạo session ID tự động
session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Tạo session ID tự động

# Khu vực người dùng upload tài liệu
uploaded_file = st.file_uploader("Tải lên tài liệu để phân tích:", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    # Gọi API upload_files bằng phương thức POST
    response = requests.post("http://localhost:8000/upload/", files={"files": uploaded_file}, data={"university_id": "your_university_id"})  # Thay thế "your_university_id" bằng ID trường thực tế
    if response.status_code == 200:
        st.success("Tải lên thành công!")
    else:
        st.error("Có lỗi xảy ra khi tải lên tài liệu.")

# Lịch sử câu hỏi và câu trả lời
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Hiển thị lịch sử chat
for question, answer in st.session_state.chat_history:
    st.write(f"**Bạn:** {question}")
    st.write(f"**Chatbot:** {answer}")

# Khu vực người dùng nhập câu hỏi
user_question = st.text_input("", placeholder="Bạn có câu hỏi gì về tuyển sinh?", key="user_input", label_visibility="collapsed")

if user_question:
    # Gọi hàm get_answer từ module chatbot.py để lấy câu trả lời
    response = get_answer(user_question, session_id=session_id)

    # Lưu câu hỏi và câu trả lời vào lịch sử
    st.session_state.chat_history.append((user_question, response))

    # Hiển thị câu trả lời
    st.write(f"**Chatbot:** {response}")


