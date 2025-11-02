# api/index.py
from backend.main import app
from mangum import Mangum

# Mangum adapts ASGI (FastAPI) to AWS Lambda-like environments used by Vercel
handler = Mangum(app)
