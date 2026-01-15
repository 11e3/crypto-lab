"""Web application configuration.

Pydantic Settings 기반 웹 애플리케이션 설정.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["WebAppSettings", "get_web_settings"]


class WebAppSettings(BaseSettings):
    """웹 애플리케이션 설정.

    환경 변수 또는 .env 파일에서 설정을 로드합니다.
    """

    model_config = SettingsConfigDict(
        env_prefix="WEB_",
        case_sensitive=False,
        extra="ignore",  # 추가 필드 무시
    )

    # Streamlit 서버 설정
    server_port: int = Field(default=8501, description="Streamlit server port")
    server_address: str = Field(default="localhost", description="Server address")
    server_headless: bool = Field(default=False, description="Run in headless mode")

    # 캐시 설정
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds (1 hour)")
    enable_caching: bool = Field(default=True, description="Enable caching")

    # UI 설정
    default_theme: str = Field(default="light", description="Default UI theme (light/dark)")
    show_debug_info: bool = Field(default=False, description="Show debug information")

    # 백테스트 설정
    max_parallel_workers: int = Field(
        default=4, description="Maximum parallel workers for optimization"
    )
    default_initial_capital: float = Field(
        default=10_000_000.0, description="Default initial capital (KRW)"
    )
    default_fee_rate: float = Field(default=0.0005, description="Default fee rate (0.05%)")
    default_slippage_rate: float = Field(
        default=0.0005, description="Default slippage rate (0.05%)"
    )


# 싱글톤 인스턴스
_settings: WebAppSettings | None = None


def get_web_settings() -> WebAppSettings:
    """웹 설정 싱글톤 인스턴스 반환."""
    global _settings
    if _settings is None:
        _settings = WebAppSettings()
    return _settings
