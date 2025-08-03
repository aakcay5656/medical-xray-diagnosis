import streamlit as st
import requests
from PIL import Image
import json
from config import BASE_URL
import time

st.set_page_config(
    page_title="Medikal AI Asistan",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    max-width: 80%;
}
.user-message {
    background-color: #e3f2fd;
    margin-left: auto;
}
.assistant-message {
    background-color: #f5f5f5;
    margin-right: auto;
}
.diagnosis-box {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
    margin: 1rem 0;
}
.error-box {
    background-color: #ffebee;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #f44336;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Session state değişkenlerini başlatır"""
    defaults = {
        "current_chat": None,
        "chat_list": [],
        "chat_messages": [],
        "diagnosis": None,
        "last_question": "",
        "upload_key": 0
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def make_request(method, endpoint, **kwargs):
    """HTTP istekleri için yardımcı fonksiyon"""
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.request(method, url, timeout=30, **kwargs)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Bağlantı hatası: {str(e)}")
        return None


def load_chat_list():
    """Chat listesini backend'den yükler"""
    response = make_request("GET", "/chat/list")
    if response and response.status_code == 200:
        st.session_state.chat_list = response.json().get("chats", [])
    else:
        st.session_state.chat_list = []


def load_chat_messages(chat_id):
    """Belirli bir chat'in mesajlarını yükler"""
    response = make_request("GET", f"/chat/{chat_id}")
    if response and response.status_code == 200:
        return response.json().get("messages", [])
    return []


def create_new_chat():
    """Yeni chat oturumu oluşturur"""
    response = make_request("POST", "/chat/new")
    if response and response.status_code == 200:
        new_chat_id = response.json()["chat_id"]
        st.session_state.current_chat = new_chat_id
        st.session_state.chat_messages = []
        st.session_state.diagnosis = None
        load_chat_list()  # Listeyi güncelle
        return True
    return False


def save_message(chat_id, question, response):
    """Mesajı backend'e kaydeder"""
    data = {
        "chat_id": chat_id,
        "question": question,
        "response": response
    }

    response = make_request("POST", "/chat/save_message", json=data)
    return response and response.status_code == 200


def delete_chat(chat_id):
    """Chat'i siler"""
    response = make_request("DELETE", f"/chat/{chat_id}")
    return response and response.status_code == 200


with st.sidebar:
    st.title("💬 Sohbetler")

    # Refresh button
    if st.button("🔄 Yenile"):
        load_chat_list()
        st.rerun()

    if st.button("➕ Yeni Sohbet", type="primary"):
        if create_new_chat():
            st.success("Yeni sohbet oluşturuldu!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Yeni sohbet oluşturulamadı.")

    st.divider()

    load_chat_list()

    if st.session_state.chat_list:
        st.subheader("Mevcut Sohbetler")

        for i, chat_id in enumerate(st.session_state.chat_list):
            col1, col2 = st.columns([3, 1])

            with col1:
                # Chat selection button
                is_current = chat_id == st.session_state.current_chat
                button_type = "primary" if is_current else "secondary"

                if st.button(
                        f"💬 {chat_id[:8]}...",
                        key=f"chat_{i}",
                        type=button_type
                ):
                    st.session_state.current_chat = chat_id
                    st.session_state.chat_messages = load_chat_messages(chat_id)
                    st.session_state.diagnosis = None
                    st.rerun()

            with col2:
                # Delete button
                if st.button("🗑️", key=f"delete_{i}", help="Sohbeti sil"):
                    if delete_chat(chat_id):
                        if chat_id == st.session_state.current_chat:
                            st.session_state.current_chat = None
                            st.session_state.chat_messages = []
                            st.rerun()
                        load_chat_list()
                        st.success("Sohbet silindi!")
                        time.sleep(0.5)
                        st.rerun()
    else:
        st.info("Henüz sohbet yok. Yeni bir sohbet başlatın!")

st.title("🩺 Medikal AI Asistan")

if not st.session_state.current_chat:
    st.warning("👈 Lütfen yan menüden bir sohbet seçin veya yeni sohbet oluşturun.")
    st.stop()

if st.session_state.diagnosis:
    st.markdown(f"""
    <div class="diagnosis-box">
        <h4>🎯 Mevcut Tanı</h4>
        <p><strong>{st.session_state.diagnosis}</strong></p>
        <p><em>Bu tanı hakkında sorular sorabilirsiniz.</em></p>
    </div>
    """, unsafe_allow_html=True)

st.subheader("💬 Sohbet Geçmişi")

if st.session_state.chat_messages:
    for i, msg in enumerate(st.session_state.chat_messages):
        # Kullanıcı mesajı
        user_html = f"""
        <div class="chat-message user-message">
            <strong>👤 Siz:</strong> {msg['question']}
        </div>
        """
        st.markdown(user_html, unsafe_allow_html=True)

        # Asistan mesajı
        assistant_html = f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Asistan:</strong> {msg['response']}
        </div>
        """
        st.markdown(assistant_html, unsafe_allow_html=True)
else:
    st.info("Henüz mesaj yok. Aşağıdan soru sorun!")

st.subheader("❓ Soru Sor")

with st.form("question_form"):
    question = st.text_area(
        "Sorunuzu yazın:",
        placeholder="Örn: Bu hastalık nasıl tedavi edilir?",
        height=100,
        key="question_input"
    )

    col1, col2 = st.columns(2)

    with col1:
        submit_button = st.form_submit_button("📤 Gönder", type="primary")

    with col2:
        clear_diagnosis = st.form_submit_button("🔄 Tanıyı Temizle")

if clear_diagnosis:
    st.session_state.diagnosis = None
    st.rerun()

if submit_button and question.strip():
    with st.spinner("🤔 Yanıt hazırlanıyor..."):
        if st.session_state.diagnosis:
            endpoint = "/ask"
            data = {
                "diagnosis": st.session_state.diagnosis,
                "question": question
            }
        else:
            endpoint = "/just_ask"
            data = {"question": question}

        response = make_request("POST", endpoint, json=data)

        if response and response.status_code == 200:
            answer = response.json()["response"]

            if save_message(st.session_state.current_chat, question, answer):
                st.session_state.chat_messages.append({
                    "question": question,
                    "response": answer
                })
                st.success("✅ Yanıt alındı!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Mesaj kaydedilemedi.")
        else:
            error_msg = "Yanıt alınamadı."
            if response:
                try:
                    error_detail = response.json().get("detail", "")
                    if error_detail:
                        error_msg += f" Hata: {error_detail}"
                except:
                    pass
            st.error(error_msg)

st.divider()
st.subheader("📷 Görüntü Analizi")

col1, col2 = st.columns([2, 1])

with col1:
    image_type = st.selectbox(
        "Görüntü Türü",
        ["akciğer", "beyin"],
        format_func=lambda x: x.capitalize(),
        help="Analiz edilecek görüntü türünü seçin"
    )

    uploaded_file = st.file_uploader(
        "MRI / Röntgen görüntüsü yükleyin",
        type=["jpg", "jpeg", "png"],
        key=f"file_upload_{st.session_state.upload_key}",
        help="Desteklenen formatlar: JPG, JPEG, PNG"
    )

with col2:
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Yüklenen Görüntü",
                use_column_width=True
            )

            # File info
            st.info(f"""
            **Dosya Bilgileri:**
            - İsim: {uploaded_file.name}
            - Boyut: {len(uploaded_file.getvalue()) / 1024:.1f} KB
            - Tip: {image_type.capitalize()}
            """)

        except Exception as e:
            st.error(f"Görüntü yüklenirken hata: {str(e)}")
            uploaded_file = None

if uploaded_file is not None:
    if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
        with st.spinner("🧠 Görüntü analiz ediliyor..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
                data = {"image_type": image_type}

                response = make_request("POST", "/predict", files=files, data=data)

                if response and response.status_code == 200:
                    result = response.json()
                    diagnosis = result["diagnosis"]

                    st.session_state.diagnosis = diagnosis
                    st.session_state.upload_key += 1

                    st.success("✅ Analiz tamamlandı!")

                    st.markdown(f"""
                    <div class="diagnosis-box">
                        <h4>🎯 Analiz Sonucu</h4>
                        <p><strong>{diagnosis}</strong></p>
                        <p><em>Bu tanı hakkında sorular sorabilirsiniz.</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                    time.sleep(1)
                    st.rerun()

                else:
                    error_msg = "Analiz sırasında hata oluştu."
                    if response:
                        try:
                            error_detail = response.json().get("detail", "")
                            if error_detail:
                                error_msg += f" Hata: {error_detail}"
                        except:
                            pass

                    st.markdown(f"""
                    <div class="error-box">
                        <h4>❌ Hata</h4>
                        <p>{error_msg}</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Beklenmeyen hata: {str(e)}")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🩺 Medikal AI Asistan | ⚠️ Bu uygulama sadece bilgilendirme amaçlıdır. Gerçek tıbbi tanı için doktorunuza başvurun.</p>
</div>
""", unsafe_allow_html=True)