from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.utilities import SerpAPIWrapper
from keybert import KeyBERT
from pinecone import Pinecone as PineconeClient
import pinecone
import logging
from typing import List, Optional
import os
import json
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cấu hình
GOOGLE_API_KEY = "AIzaSyBYPplSIEGH_Rq3K4RYDMovbnjnf-5zHU4"
SERP_API_KEY = "062caddf2fd2f14c079e2a99e38afc4b3733ba36b9afa1d755d8cf202db6f469"
PINECONE_API_KEY = "pcsk_4PC5Nf_Pgh2N5m8hdN3BgvqPF4s9xaDxJZ6HYyXZRYSi1FdnqLYSRbJV4jFAw8zSULdcRD"
PINECONE_ENV = "us-east-1"
LOCAL_DATA_DIR = "data/documents"  # Thư mục chứa tài liệu
SESSION_DIR = "data/sessions"  # Thư mục lưu session

# Thay đổi cách khởi tạo Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Thay đổi tên index để khớp với index đã tạo
INDEX_NAME = "gemini-embeddings"



# System prompt
SYSTEM_PROMPT = """Bạn là một trợ lý tư vấn tuyển sinh thông minh. Khi trả lời:
1. Hãy tóm tắt thông tin một cách ngắn gọn, dễ hiểu
2. Trình bày thông tin theo dạng có cấu trúc với các bullet points
3. Nếu thông tin từ web, hãy trích dẫn nguồn và đưa vào dạng link
4. Tập trung vào những thông tin quan trọng và liên quan nhất
5. Sử dụng ngôn ngữ thân thiện, dễ hiểu và cùng ngôn ngữ với ngôn ngữ của câu hỏi người dùng đưa ra

Ví dụ format câu trả lời:
- Điểm 1
- Điểm 2
- Điểm 3

Nguồn: [link hoặc tên website]
"""

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []
        self.session_file = os.path.join(SESSION_DIR, f"{session_id}.json")
        self.load_session()

    def load_session(self):
        if os.path.exists(self.session_file):
            with open(self.session_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)

    def save_session(self):
        os.makedirs(SESSION_DIR, exist_ok=True)
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add_interaction(self, question: str, answer: str):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })
        if len(self.history) > 5:  # Giữ lại 5 tương tác gần nhất
            self.history = self.history[-5:]
        self.save_session()

    def get_history_str(self) -> str:
        return "\n".join([
            f"Q: {interaction['question']}\nA: {interaction['answer']}"
            for interaction in self.history
        ])

def load_local_documents():
    documents = []
    for root, _, files in os.walk(LOCAL_DATA_DIR):
        for file in files:
            if file.endswith(('.txt', '.pdf', '.docx')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logging.info(f"Đã đọc file: {file_path}")  # Thêm logging
                        documents.append({
                            "content": content,
                            "source": file_path
                        })
                except Exception as e:
                    logging.error(f"Lỗi khi đọc file {file_path}: {e}")
    logging.info(f"Tổng số documents đã đọc: {len(documents)}")  # Thêm logging
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store(texts):
    if not texts:  # Thêm kiểm tra
        logging.error("Không có texts để upload")
        return None
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    logging.info(f"Bắt đầu upload {len(texts)} documents vào index {INDEX_NAME}")
    
    try:
        # Xóa dữ liệu cũ trong index (nếu cần)
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True)
        
        vector_store = Pinecone.from_documents(
            texts,
            embeddings,
            index_name=INDEX_NAME
        )
        logging.info("Upload documents thành công")
        return vector_store
    except Exception as e:
        logging.error(f"Lỗi khi upload documents: {e}")
        raise

def extract_keywords(question: str, top_n: int = 3) -> List[str]:
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

def search_on_web(query: str) -> Optional[str]:
    try:
        search = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)
        results = search.run(query)
        return results
    except Exception as e:
        logging.error(f"Lỗi khi tìm kiếm web: {e}")
        return None

def format_web_response(web_result: str, query: str, history: str = "") -> str:
    try:
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        prompt = f"""
        {SYSTEM_PROMPT}
        
        Lịch sử hội thoại:
        {history}
        
        Câu hỏi hiện tại: "{query}"
        Thông tin tìm được: "{web_result}"
        
        Hãy format lại câu trả lời theo hướng dẫn trên.
        """
        
        formatted_response = llm.invoke(prompt)
        return formatted_response
    except Exception as e:
        logging.error(f"Lỗi khi format câu trả lời: {e}")
        return web_result

def initialize_chatbot(vector_store):
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    chatbot = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    
    return chatbot

def chat(session_id: str = None):
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session = ChatSession(session_id)
    
    # Tải và xử lý dữ liệu local
    documents = load_local_documents()
    if not documents:
        print("Không có dữ liệu nào được tải. Vui lòng kiểm tra thư mục data/documents")
        return

    texts = split_text(documents)
    vector_store = create_vector_store(texts)
    chatbot = initialize_chatbot(vector_store)

    print("Chatbot: Xin chào! Bạn có thể đặt câu hỏi về tuyển sinh.")
    
    while True:
        try:
            user_input = input("Bạn: ").strip()
            if not user_input:
                print("Chatbot: Vui lòng nhập câu hỏi của bạn.")
                continue
                
            if user_input.lower() in ["e", "q", "tạm biệt", "exit", "quit"]:
                print("Chatbot: Tạm biệt! Hẹn gặp lại.")
                break

            # Lấy lịch sử hội thoại
            history = session.get_history_str()
            
            # Trích xuất từ khóa từ câu hỏi
            keywords = extract_keywords(user_input)
            if not keywords:
                print("Chatbot: Xin lỗi, tôi không hiểu câu hỏi của bạn. Vui lòng thử lại.")
                continue
            
            # Tìm kiếm trong vector store
            response = chatbot.invoke({"query": user_input})
            
            # Kiểm tra chất lượng câu trả lời và tìm kiếm web nếu cần
            if (not response or 
                len(str(response).strip()) < 50 or 
                "không tìm thấy thông tin" in str(response).lower() or
                "tôi không biết" in str(response).lower()):
                
                logging.info("Không tìm thấy thông tin trong context, đang tìm kiếm web...")
                print("Chatbot: Đang tìm kiếm thông tin bổ sung trên web...")
                
                search_query = f"tuyển sinh đại học {' '.join(keywords)}"
                web_result = search_on_web(search_query)
                
                if web_result:
                    response = format_web_response(web_result, user_input, history)
                else:
                    response = "Xin lỗi, tôi không tìm thấy thông tin liên quan cả trong cơ sở dữ liệu lẫn trên web."

            response_text = str(response)
            print(f"Chatbot: {response_text}")
            
            # Lưu tương tác vào session
            session.add_interaction(user_input, response_text)
            
        except KeyboardInterrupt:
            print("\nChatbot: Tạm biệt! Hẹn gặp lại.")
            break
        except Exception as e:
            logging.error(f"Lỗi không mong muốn: {e}")
            print("Chatbot: Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.")

def check_pinecone_index(index_name: str) -> bool:
    """
    Kiểm tra xem index có tồn tại trong Pinecone không
    """
    try:
        active_indexes = pc.list_indexes().names()
        return index_name in active_indexes
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra Pinecone index: {e}")
        return False

if __name__ == "__main__":
    chat() 