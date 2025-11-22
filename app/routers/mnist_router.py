from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.predict import ResponseOutput
from app.utils.preprocessing import preprocess_image
from app.models.xgb_model import model

router = APIRouter(prefix="/predict", tags=["predict"])

ALLOWED_FORMATS = ["image/jpeg", "image/jpg", "image/png"]
ALOWED_SIZE = 1024 * 1024 * 2

@router.post("/", response_model=ResponseOutput)
async def predict_digit(file : UploadFile = File(...,
                             description="Upload a file",
                             example="digit_7.jpg",)):

    if file.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Allowed: {ALLOWED_FORMATS}"
        )

    image = await file.read()

    if len(image) > ALOWED_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"file size exceeds 2m limits"
        )


    features = preprocess_image(image)
    features = features.reshape(1, -1)
    digit = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]

    return ResponseOutput(
        predicted_digit=digit,
        confidence=float(probs.max()),
        probability={str(i): float(p) for i, p in enumerate(probs)}
    )


