# risk/apps.py
from django.apps import AppConfig
import logging
import os

logger = logging.getLogger(__name__)

class RiskConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'risk'

    def ready(self):
        # Prevent startup side effects on Render/Gunicorn
        render_flag = os.getenv("RENDER", "").strip().lower()
        if render_flag in {"1", "true", "yes"}:
            logger.info("Skipping MCP startup on Render.")
            return

        try:
            from risk.services import get_global_agent
            get_global_agent()
            logger.info("MCP agent pre-started.")
        except Exception:
            logger.exception("Failed to start MCP agent")