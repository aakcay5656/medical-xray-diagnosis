from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI


from app.rag.query_rag import ask_with_context_lung, ask_with_context_brain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not found")

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        max_output_tokens=1000,
        timeout=30,
        convert_system_message_to_human=True,

    )
    logger.info("Gemini LLM successfully initialized")
except Exception as e:
    logger.error(f"Gemini LLM initialization error: {e}")
    raise


@tool
def explain_diagnosis(diagnosis: str) -> str:
    """Tanıya göre Türkçe tıbbi açıklama yapar (nedenler, belirtiler, tedaviler)."""
    if not diagnosis.strip():
        return "Geçerli bir tanı bilgisi sağlanmadı."

    try:
        prompt = f"""
        Sen deneyimli bir doktorsun. Bir hastaya '{diagnosis}' teşhisi kondu.
        Bu hastalık hakkında hasta ve yakınlarının anlayabileceği şekilde açıklama yap:

        1. Hastalık nedir?
        2. Ana nedenleri nelerdir?
        3. Tipik belirtileri nelerdir?
        4. Tedavi seçenekleri nelerdir?
        5. Hastanın dikkat etmesi gerekenler

        Açıklamanı TÜRKÇE, anlaşılır ve empatik bir dille yap.
        Tıbbi terimler kullanırken açıklamalarını da ekle.
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error in explain_diagnosis: {e}")
        return "Tanı açıklaması sırasında bir hata oluştu."


@tool
def medical_researcher(diagnosis: str) -> str:
    """Hastalığın tarihçesini ve güncel bilimsel bilgilerini TÜRKÇE olarak açıklar."""
    if not diagnosis.strip():
        return "Geçerli bir tanı bilgisi sağlanmadı."

    try:
        prompt = f"""
        Sen tıbbi literatürde uzman bir araştırmacısın. '{diagnosis}' hastalığı hakkında:

        1. Hastalığın tarihçesi ve keşfi
        2. Epidemiyolojik bilgiler (sıklık, risk faktörleri)
        3. Patofizyoloji (hastalık mekanizması)
        4. Güncel tanı yöntemleri
        5. Son yıllardaki tedavi gelişmeleri
        6. Prognoz (hastalığın gidişatı)

        Bu bilgileri TÜRKÇE, bilimsel ama anlaşılır bir dille sun.
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error in medical_researcher: {e}")
        return "Araştırma bilgileri alınırken bir hata oluştu."


@tool
def pulmonology_expert(question: str) -> str:
    """Akciğer hastalıkları uzmanı olarak TÜRKÇE tıbbi soruları yanıtlar."""
    if not question.strip():
        return "Geçerli bir soru sağlanmadı."

    try:
        prompt = f"""
        Sen deneyimli bir göğüs hastalıkları uzmanısın. 
        Aşağıdaki soruyu akciğer hastalıkları perspektifinden yanıtla:

        Soru: {question}

        Yanıtını TÜRKÇE ver ve:
        - Tıbbi açıklamalarını hasta anlayacak düzeyde yap
        - Gerekirse teşhis ve tedavi önerilerinde bulun
        - Risk faktörlerini ve önlemleri belirt
        - Şüphe durumunda doktora başvurulması gerektiğini belirt
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error in pulmonology_expert: {e}")
        return "Göğüs hastalıkları uzmanı yanıtı alınırken hata oluştu."


@tool
def neurology_expert(question: str) -> str:
    """Nöroloji uzmanı olarak beyin ve sinir sistemi hastalıklarıyla ilgili TÜRKÇE soruları yanıtlar."""
    if not question.strip():
        return "Geçerli bir soru sağlanmadı."

    try:
        prompt = f"""
        Sen deneyimli bir nöroloji uzmanısın.
        Aşağıdaki soruyu beyin ve sinir sistemi hastalıkları perspektifinden yanıtla:

        Soru: {question}

        Yanıtını TÜRKÇE ver ve:
        - Nörolojik açıklamalarını hasta anlayacak düzeyde yap
        - Semptomları ve tanı süreçlerini açıkla
        - Tedavi seçeneklerini belirt
        - Acil durumları belirt
        - Uzman hekime başvuru durumlarını açıkla
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error in neurology_expert: {e}")
        return "Nöroloji uzmanı yanıtı alınırken hata oluştu."


@tool
def lung_knowledge_base(question: str) -> str:
    """Akciğer hastalıkları bilgi tabanından TÜRKÇE yanıt verir."""
    if not question.strip():
        return "Geçerli bir soru sağlanmadı."

    try:
        return ask_with_context_lung(question)
    except Exception as e:
        logger.error(f"Error in lung_knowledge_base: {e}")
        return "Akciğer hastalıkları bilgi tabanından yanıt alınırken hata oluştu."


@tool
def brain_knowledge_base(question: str) -> str:
    """Beyin hastalıkları bilgi tabanından TÜRKÇE yanıt verir."""
    if not question.strip():
        return "Geçerli bir soru sağlanmadı."

    try:
        return ask_with_context_brain(question)
    except Exception as e:
        logger.error(f"Error in brain_knowledge_base: {e}")
        return "Beyin hastalıkları bilgi tabanından yanıt alınırken hata oluştu."


# === SYSTEM PROMPT ===
system_prompt = """
Sen deneyimli bir tıbbi AI asistanısın. Görevin hastalara ve yakınlarına tıbbi konularda yardımcı olmak.

İLKELERİN:
1. Her zaman TÜRKÇE yanıt ver
2. Tıbbi bilgileri anlaşılır ve empatik bir dille sun
3. Kesin tanı koymaktan kaçın, sadece bilgilendirme yap  
4. Acil durumlarda derhal doktora başvurulmasını öner
5. Şüphe durumlarında uzman hekime yönlendir
6. Tedavi önerilerinde "doktorunuzla görüşün" ifadesini kullan

YAPISAL YAKLAŞIMIN:
- Önce hastanın sorusunu tam olarak anla
- Gerekirse uygun tool'ları kullan
- Bilgileri hasta odaklı bir şekilde düzenle
- Net ve yararlı yanıt ver

UYARILAR:
- Kesinlikle tanı koyma
- İlaç dozaj önerisi verme
- Doktor yerine geçmeye çalışma
- Panik yaratacak ifadeler kullanma

Hastalar seninle konuştuğunda kendilerini güvende hissetmeli.
"""

# === PROMPT TEMPLATE ===
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10,
    output_key="output"
)

tools = [
    explain_diagnosis,
    medical_researcher,
    pulmonology_expert,
    neurology_expert,
    lung_knowledge_base,
    brain_knowledge_base
]


def create_medical_agent():
    """Tıbbi AI agent'ını oluşturur"""
    try:
        agent = create_openai_tools_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_template
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=60
        )

        logger.info("Medical AI Agent successfully created")
        return agent_executor

    except Exception as e:
        logger.error(f"Error creating medical agent: {e}")
        raise


agent_executor = create_medical_agent()


def ask_medical_question(question: str, diagnosis: str = None) -> str:
    """Tıbbi soru sorma fonksiyonu"""
    try:
        if diagnosis:
            full_question = f"Tanı: {diagnosis}\nSoru: {question}"
        else:
            full_question = question

        response = agent_executor.invoke({"input": full_question})
        return response.get("output", "Yanıt alınamadı.")

    except Exception as e:
        logger.error(f"Error in ask_medical_question: {e}")
        return "Üzgünüm, şu anda sorunuzu yanıtlayamıyorum. Lütfen daha sonra tekrar deneyin."


def clear_conversation_history():
    """Konuşma geçmişini temizler"""
    try:
        memory.clear()
        logger.info("Conversation history cleared")
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")


if __name__ == "__main__":
    try:
        diagnosis = "Zatürre"
        question = "Bu hastalık bulaşıcı mı?"

        print("🩺 Tıbbi AI Asistan Test Ediliyor...")
        print(f"Tanı: {diagnosis}")
        print(f"Soru: {question}")
        print("=" * 50)

        response = ask_medical_question(question, diagnosis)
        print("Yanıt:")
        print(response)

        print("\n" + "=" * 50)
        general_question = "Migren ağrısı nasıl geçer?"
        print(f"Genel Soru: {general_question}")
        print("=" * 50)

        response2 = ask_medical_question(general_question)
        print("Yanıt:")
        print(response2)

    except Exception as e:
        print(f"Test sırasında hata: {e}")