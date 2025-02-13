from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_community import GoogleDriveLoader
from google.oauth2 import service_account
from langchain.utilities import SerpAPIWrapper
from keybert import KeyBERT
import logging
from typing import List, Optional

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cấu hình
folder_id = "1JB5TXVjkc4xr7LK7HEkz999NS8GNDkQE"
service_account_file = 'C:/Users/nguyen/Downloads/n8n-tuyensinhdaihoc-31d954adc7bb.json'
google_api_key = "AIzaSyBYPplSIEGH_Rq3K4RYDMovbnjnf-5zHU4"
serp_api_key = "062caddf2fd2f14c079e2a99e38afc4b3733ba36b9afa1d755d8cf202db6f469"

# Thêm system prompt
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

# Tải dữ liệu từ Google Drive
def load_files_from_drive(folder_id):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            credentials=credentials,
            file_types=["document", "pdf", "sheet"]  
        )
        documents = loader.load()
        return documents
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu từ Google Drive: {e}")
        return []

# Chia nhỏ văn bản
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Tạo vector store
def create_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    return FAISS.from_documents(texts, embeddings)

# Trích xuất từ khóa quan trọng từ câu hỏi
def extract_keywords(question: str, top_n: int = 3) -> List[str]:
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

# Tìm kiếm thông tin trên SerpAPI
def search_on_web(query: str) -> Optional[str]:
    try:
        search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
        results = search.run(query)
        return results
    except Exception as e:
        logging.error(f"Lỗi khi tìm kiếm web: {e}")
        return None

# Khởi tạo chatbot với chức năng tìm kiếm web khi cần thiết
def initialize_chatbot(vector_store):
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    memory = ConversationBufferMemory()
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Lấy 5 kết quả gần nhất
    chatbot = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    
    return chatbot

def format_web_response(web_result: str, query: str) -> str:
    try:
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        prompt = f"""
        {SYSTEM_PROMPT}
        
        Dựa trên câu hỏi: "{query}"
        Và thông tin tìm được: "{web_result}"
        
        Hãy format lại câu trả lời theo hướng dẫn trên.
        """
        
        formatted_response = llm.invoke(prompt)
        return formatted_response
    except Exception as e:
        logging.error(f"Lỗi khi format câu trả lời: {e}")
        return web_result

# Hàm chính
def chat():
    documents = load_files_from_drive(folder_id)
    if not documents:
        print("Không có dữ liệu nào được tải. Vui lòng kiểm tra lại.")
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

            # Trích xuất từ khóa từ câu hỏi
            keywords = extract_keywords(user_input)
            if not keywords:
                print("Chatbot: Xin lỗi, tôi không hiểu câu hỏi của bạn. Vui lòng thử lại.")
                continue
            
            search_query = " ".join(keywords)
            
            # Tìm kiếm trong vector store trước
            response = chatbot.run(user_input)
            
            # Kiểm tra chất lượng câu trả lời và tìm kiếm web nếu cần
            if (not response or 
                len(response.strip()) < 50 or 
                "không tìm thấy thông tin" in response.lower() or
                "tôi không biết" in response.lower()):
                
                # logging.info("Không tìm thấy thông tin trong context, đang tìm kiếm web...")
                # print("Chatbot: Đang tìm kiếm thông tin bổ sung trên web...")
                
                search_query = f"tuyển sinh đại học {' '.join(keywords)}"
                web_result = search_on_web(search_query)
                
                if web_result:
                    # Format lại câu trả lời từ web
                    formatted_response = format_web_response(web_result, user_input)
                    response = formatted_response
                else:
                    response = "Xin lỗi, tôi không tìm thấy thông tin liên quan cả trong cơ sở dữ liệu lẫn trên web."

            print(f"Chatbot: {response}")
            
        except KeyboardInterrupt:
            print("\nChatbot: Tạm biệt! Hẹn gặp lại.")
            break
        except Exception as e:
            logging.error(f"Lỗi không mong muốn: {e}")
            print("Chatbot: Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.")

if __name__ == "__main__":
    chat()
