# D:\스마트팜\main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io, base64, cv2, hashlib
from datetime import datetime, timezone
from fastapi import Query
MODEL_VERSION = 'YOLO11n-' + hashlib.sha256(open('best.pt', 'rb').read()).hexdigest()[:12]

app = FastAPI(title="설향 딸기 숙성도 판독 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 1회만 로드
model = YOLO("best.pt")
KOR_MAP = {"Fullripe": "완숙", "Semiripe": "반숙", "Unripe": "미숙"}


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_classes": model.names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.05, le=0.95)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")

    raw = await file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, '사진은 10MB 이하여야 합니다.')
    try:
        image = Image.open(io.BytesIO(raw))
        if image.format not in ('JPEG', 'PNG', 'WEBP') or min(image.size) < 32 or max(image.size) > 6000 or image.width * image.height > 20000000:
            raise ValueError('unsupported image')
        image = image.convert('RGB')
    except Exception:
        raise HTTPException(400, "이미지를 읽을 수 없습니다.")

    results = model(image, conf=conf, verbose=False)[0]

    counts = {"Fullripe": 0, "Semiripe": 0, "Unripe": 0}
    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name in counts:
            counts[cls_name] += 1
        detections.append({
            "class": cls_name,
            "class_kor": KOR_MAP.get(cls_name, cls_name),
            "confidence": round(float(box.conf), 4),
            "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })

    # 주석 이미지를 base64로 반환 (디스크 저장 X)
    annotated_bgr = results.plot()
    ok, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "결과 이미지 생성 실패")

    return {
        "model_version": MODEL_VERSION,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "counts_kor": {KOR_MAP[k]: v for k, v in counts.items()},
        "total": sum(counts.values()),
        "detections": detections,
        "annotated_image_base64": base64.b64encode(buf.tobytes()).decode("utf-8"),
    }


app.mount("/static", StaticFiles(directory="static"), name="static")
