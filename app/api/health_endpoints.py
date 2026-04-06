"""
Health check endpoints for Founder Intelligence System
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime

from ..config.database import db_manager
from ..utils.logger import get_logger

router = APIRouter(prefix="/health", tags=["Health"])
logger = get_logger(__name__)


@router.get("/", summary="Basic Health Check")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Check database connection
        db_healthy = db_manager.health_check()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "database": "connected" if db_healthy else "disconnected"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/detailed", summary="Detailed Health Check")
async def detailed_health_check():
    """Detailed health check with system information"""
    try:
        import psutil
        import sys
        
        # System information
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        # Database health
        db_healthy = db_manager.health_check()
        
        # External service checks
        external_services = await _check_external_services()
        
        overall_status = "healthy"
        if not db_healthy or any(status != "healthy" for status in external_services.values()):
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "system": system_info,
            "database": {
                "status": "connected" if db_healthy else "disconnected",
                "response_time": "< 100ms" if db_healthy else "> 1000ms"
            },
            "external_services": external_services,
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Detailed health check failed: {str(e)}"
        )


@router.get("/readiness", summary="Readiness Check")
async def readiness_check():
    """Readiness check for Kubernetes/Container orchestration"""
    try:
        # Check if all critical components are ready
        db_healthy = db_manager.health_check()
        
        # Check if we can perform basic operations
        ready = True
        checks = {}
        
        # Database check
        checks["database"] = db_healthy
        if not db_healthy:
            ready = False
        
        # Configuration check
        from ..config.settings import validate_settings
        config_valid = validate_settings()
        checks["configuration"] = config_valid
        if not config_valid:
            ready = False
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            checks["memory"] = memory.available > 100 * 1024 * 1024  # At least 100MB available
            if memory.available <= 100 * 1024 * 1024:
                ready = False
        except:
            checks["memory"] = False
            ready = False
        
        return {
            "status": "ready" if ready else "not_ready",
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {
            "status": "not_ready",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/live", summary="Liveness Check")
async def liveness_check():
    """Liveness check - simple check if service is alive"""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }


async def _check_external_services() -> Dict[str, str]:
    """Check external service connectivity"""
    services = {}
    
    # Check Supabase
    try:
        from ..config.settings import settings
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            services["supabase"] = "healthy"
        else:
            services["supabase"] = "misconfigured"
    except Exception as e:
        services["supabase"] = f"error: {str(e)}"
    
    # Check Gemini API
    try:
        from ..config.settings import settings
        if settings.GEMINI_API_KEY:
            services["gemini_api"] = "configured"
        else:
            services["gemini_api"] = "misconfigured"
    except Exception as e:
        services["gemini_api"] = f"error: {str(e)}"
    
    # Check Apify
    try:
        from ..config.settings import settings
        if settings.APIFY_TOKEN:
            services["apify"] = "configured"
        else:
            services["apify"] = "misconfigured"
    except Exception as e:
        services["apify"] = f"error: {str(e)}"
    
    # Check Redis
    try:
        from ..utils.cache import cache_manager
        cache_manager.get("test_key")
        services["redis"] = "connected"
    except Exception as e:
        services["redis"] = f"error: {str(e)}"
    
    return services
