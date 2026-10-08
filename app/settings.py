"""
Deployment settings, read from environment variables (or a local .env file).

Rule of thumb:
  - Things that change per ENVIRONMENT (URLs, secrets, ports)  -> here (.env)
  - Things that change per AGENT (prompt, tools, limits)       -> agent/agent.yaml
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service -------------------------------------------------------------
    app_env: Literal["dev", "staging", "production"] = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_dir: str = "logs"                 # app.log + requests/ live here; "" = no log files
    log_request_files: bool = True        # one log file per API call: logs/requests/<day>/<request_id>.log
    log_request_payloads: bool = True     # include messages, tool args/results in request files (PII! see README)
    log_retention_days: int = 14          # request files older than this are deleted at startup

    # --- LLM (any OpenAI-compatible endpoint: vLLM, OpenAI, Azure, Ollama...) ---
    llm_api_key: SecretStr = SecretStr("not-needed")
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2  # retries 429 / 5xx / connection errors with backoff

    # --- Security --------------------------------------------------------------
    # Comma-separated "caller_name:key" pairs, e.g. "web-ui:abc123,n8n:def456".
    # The caller name shows up in logs and is used for rate limiting.
    api_keys: SecretStr = SecretStr("")
    rate_limit_per_minute: int = 30
    max_concurrent_per_caller: int = 3        # 0 = unlimited
    daily_token_budget_per_caller: int = 0    # LLM tokens per caller per day, 0 = unlimited
    max_concurrent_runs: int = 20             # whole service; protects the LLM server. 0 = unlimited
    max_input_chars: int = 8000
    cors_origins: str = ""  # comma-separated; empty = CORS disabled

    # --- Sessions (conversation memory) ----------------------------------------
    session_ttl_seconds: int = 3600
    max_sessions: int = 10_000

    # --- Monitoring ----------------------------------------------------------
    metrics_enabled: bool = True

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def api_key_map(self) -> dict[str, str]:
        """{key: caller_name}"""
        result: dict[str, str] = {}
        for pair in self.api_keys.get_secret_value().split(","):
            pair = pair.strip()
            if not pair:
                continue
            name, sep, key = pair.partition(":")
            if not sep or not name or len(key) < 16:
                raise ValueError("API_KEYS entries must look like 'caller:key' with a key of 16+ chars")
            result[key] = name
        return result

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _production_safety(self) -> "Settings":
        keys = self.api_key_map  # validates format
        if self.app_env == "production" and not keys:
            raise ValueError("API_KEYS must be set when APP_ENV=production (auth cannot be disabled in production)")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
