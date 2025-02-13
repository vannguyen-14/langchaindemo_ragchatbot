from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from new_langchain_demo import (
    load_local_documents, 
    split_text, 
    create_vector_store,
    initialize_chatbot,
    check_pinecone_index
)
import logging

app = Flask(__name__, 
    template_folder='templates',
    static_folder='app/static'
)
app.secret_key = '1234567890'

# Cấu hình
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'data', 'documents')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn kích thước file 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không tìm thấy file')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        
        if not files or all(file.filename == '' for file in files):
            flash('Không có file nào được chọn')
            return redirect(request.url)
        
        # Tạo thư mục upload nếu chưa tồn tại
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    uploaded_files.append(filename)
                    flash(f'Đã upload thành công file: {filename}')
                except Exception as e:
                    flash(f'Lỗi khi lưu file {filename}: {str(e)}')
                    continue
        
        if uploaded_files:
            try:
                # Cập nhật vector store
                documents = load_local_documents()
                if documents:
                    logging.info(f"Đã đọc được {len(documents)} tài liệu")
                    texts = split_text(documents)
                    if texts:
                        logging.info(f"Đã tách được {len(texts)} đoạn văn bản")
                        vector_store = create_vector_store(texts)
                        
                        # Kiểm tra index sau khi tạo
                        if check_pinecone_index():
                            flash('Đã cập nhật thành công vector store')
                        else:
                            flash('Đã upload file nhưng có vấn đề với vector store')
                    else:
                        flash('Không thể tách văn bản từ tài liệu')
                else:
                    flash('Không thể đọc dữ liệu từ các file đã upload')
            except Exception as e:
                flash(f'Lỗi khi cập nhật vector store: {str(e)}')
                logging.error(f'Lỗi chi tiết: {str(e)}')
        
        return redirect(url_for('documents'))
    
    return render_template('upload.html')

@app.route('/documents')
def documents():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_stats = os.stat(file_path)
            files.append({
                'name': filename,
                'size': file_stats.st_size,
                'modified': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    return render_template('documents.html', files=files)

@app.route('/delete/<filename>')
def delete_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            flash(f'File {filename} đã được xóa')
            
            # Cập nhật lại vector store
            documents = load_local_documents()
            texts = split_text(documents)
            vector_store = create_vector_store(texts)
        else:
            flash(f'File {filename} không tồn tại')
    except Exception as e:
        flash(f'Lỗi khi xóa file: {str(e)}')
    
    return redirect(url_for('documents'))

if __name__ == '__main__':
    # Đảm bảo thư mục upload tồn tại
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
