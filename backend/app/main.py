import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.routes_detect import router as detect_router
from app.security import ensure_auth_users_bootstrapped

load_dotenv()

cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["http://localhost:5173"]

app = FastAPI(title="DeepShield AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router, prefix="/api/v1")


@app.on_event("startup")
def startup_seed_auth_users():
    # Ensure default admin/user accounts exist in MongoDB before readiness checks.
    ensure_auth_users_bootstrapped()


@app.get("/")
def root():
    return {"message": "DeepShield AI backend is running"}
