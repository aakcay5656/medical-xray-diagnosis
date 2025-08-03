from fastapi import FastAPI, UploadFile, File, Form, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import traceback
import logging

from app.api.schemas import SaveMessageRequest,AskRequest,JustAskRequest

from app.inference.predict_diagnosis import (
    predict_lung_from_bytes,
    predict_brain_from_bytes,
    load_model_lung,
    load_model_brain
)
from app.agents.langchainagent import agent_executor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = FastAPI(
    title="Medical Diagnosis API",
    description="Tıbbi tanı ve sohbet API'si",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    models = {
        "akciğer": load_model_lung(),
        "beyin": load_model_brain()
    }
    logger.info("Modeller başarıyla yüklendi")
except Exception as e:
    logger.error(f"Model yükleme hatası: {e}")
    models = {}

predict_funcs = {
    "akciğer": predict_lung_from_bytes,
    "beyin": predict_brain_from_bytes
}


chat_data = {}  # {"chat_id": [{"question": "...", "response": "..."}]}


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/new")
def new_chat():
    """Yeni bir chat oturumu oluşturur"""
    chat_id = str(uuid.uuid4())
    chat_data[chat_id] = []
    logger.info(f"Yeni chat oluşturuldu: {chat_id}")
    return {"chat_id": chat_id}


@router.post("/save_message")
def save_message(request: SaveMessageRequest):
    """Chat oturumuna mesaj kaydeder"""
    if request.chat_id not in chat_data:
        raise HTTPException(status_code=404, detail="Chat bulunamadı")

    chat_data[request.chat_id].append({
        "question": request.question,
        "response": request.response
    })
    logger.info(f"Mesaj kaydedildi - Chat ID: {request.chat_id}")
    return {"status": "success"}


@router.get("/list")
def get_chats():
    """Tüm chat ID'lerini listeler"""
    return {"chats": list(chat_data.keys())}


@router.get("/{chat_id}")
def get_chat(chat_id: str):
    """Belirli bir chat'in mesajlarını getirir"""
    if chat_id not in chat_data:
        raise HTTPException(status_code=404, detail="Chat bulunamadı")
    return {"messages": chat_data[chat_id]}


@router.delete("/{chat_id}")
def delete_chat(chat_id: str):
    """Bir chat oturumunu siler"""
    if chat_id not in chat_data:
        raise HTTPException(status_code=404, detail="Chat bulunamadı")

    del chat_data[chat_id]
    logger.info(f"Chat silindi: {chat_id}")
    return {"status": "success", "message": "Chat silindi"}


app.include_router(router)



@app.post("/predict", tags=["Prediction"])
async def predict_endpoint(file: UploadFile = File(...), image_type: str = Form(...)):
    """Tıbbi görüntü analizi yapar"""
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Geçersiz dosya türü. Lütfen bir görüntü dosyası yükleyin.")

        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="Dosya çok büyük. Maksimum 10MB.")

        if image_type not in models:
            available_types = list(models.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Geçersiz görüntü türü. Desteklenen türler: {available_types}"
            )


        model = models[image_type]
        if model is None:
            raise HTTPException(status_code=500, detail=f"{image_type} modeli yüklenemedi")


        predict_fn = predict_funcs[image_type]
        prediction = predict_fn(contents, model)

        logger.info(f"Tahmin yapıldı - Tür: {image_type}, Sonuç: {prediction}")
        return JSONResponse(content={
            "diagnosis": prediction,
            "type": image_type,
            "filename": file.filename
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Tahmin işlemi sırasında bir hata oluştu")



@app.post("/ask", tags=["Q&A"])
async def ask_with_diagnosis(request: AskRequest):
    """Tanı bilgisi ile soru sorar"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Soru boş olamaz")

        if not request.diagnosis.strip():
            raise HTTPException(status_code=400, detail="Tanı bilgisi boş olamaz")

        prompt = f"Yanıtlar Türkçe olarak verilecek. Tanı: {request.diagnosis}, Soru: {request.question}"
        response = agent_executor.invoke({"input": prompt})


        if isinstance(response, dict) and "output" in response:
            response = response["output"]
        else:
            response = str(response)

        logger.info(f"Tanı ile soru yanıtlandı - Tanı: {request.diagnosis[:50]}...")
        return JSONResponse(content={"response": response})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Soru-cevap hatası: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Soru yanıtlanırken bir hata oluştu")



@app.post("/just_ask", tags=["Q&A"])
async def ask_without_diagnosis(request: JustAskRequest):
    """Tanı bilgisi olmadan soru sorar"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Soru boş olamaz")

        prompt = f"Yanıtlar Türkçe olarak verilecek. Soru: {request.question}"
        agent_response = agent_executor.invoke({"input": prompt})


        if isinstance(agent_response, dict) and "output" in agent_response:
            response = agent_response["output"]
        else:
            response = str(agent_response)

        logger.info("Genel soru yanıtlandı")
        return JSONResponse(content={"response": response})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Soru-cevap hatası: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Soru yanıtlanırken bir hata oluştu")



@app.get("/health", tags=["Health"])
def health_check():
    """API'nin sağlık durumunu kontrol eder"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "active_chats": len(chat_data)
    }