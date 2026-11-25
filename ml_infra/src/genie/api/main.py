"""FastAPI application entrypoint for Genie Stateless ML Service."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from genie.api.v1.discovery import router as discovery_router
from genie.api.v1.verify import router as verify_router
from genie.common.config import settings
from genie.common.exceptions import QualityGateError


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models and warm caches
    yield
    # Cleanup


app = FastAPI(
    title="GENIE AI Stateless ML Verification Service",
    description="Pure ML & computer vision verification layer powering Genium (Backend owns PostgreSQL).",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(QualityGateError)
async def quality_gate_exception_handler(request: Request, exc: QualityGateError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": exc.code,
            "message": str(exc),
            "details": exc.details,
        },
    )


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "genie-ml-infra",
        "mode": "stateless",
        "env": settings.ENV,
    }


app.include_router(verify_router, prefix="/v1")
app.include_router(discovery_router, prefix="/v1")
