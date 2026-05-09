# risk/apps.py
from django.apps import AppConfig
import logging
from risk.services import get_global_agent

class RiskConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'risk'

    def ready(self):
        # Pre‑start the MCP agent when Django starts
        get_global_agent()
        print("✅ MCP agent pre‑started.")