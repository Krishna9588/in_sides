"""
Main FastAPI application for Founder Intelligence System
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from .config.settings import settings, validate_settings
from .agents.agent_1 import ResearchIngestionAgent
from .utils.logger import main_logger
from .api import agent_endpoints, chat_endpoints, health_endpoints

# Validate settings
if not validate_settings():
    main_logger.error("Invalid configuration. Please check your environment variables.")
    exit(1)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered decision intelligence platform for founders",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
agent_1 = ResearchIngestionAgent()

# Include API routers
app.include_router(health_endpoints.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(agent_endpoints.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(chat_endpoints.router, prefix="/api/v1/chat", tags=["Chat"])


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    main_logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    main_logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    main_logger.info("Application shutdown")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    main_logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
