"""Web configuration package."""

from src.web.config.app_settings import WebAppSettings, get_web_settings

__all__ = [
    "WebAppSettings",
    "get_web_settings",
]
