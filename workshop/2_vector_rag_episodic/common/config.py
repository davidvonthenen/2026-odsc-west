"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import platform
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    """Runtime configuration parameters for the episodic-memory RAG workflow."""

    # OpenSearch
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_index: str = "bbc-vector-chunks"

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # which setting
    external_ai: bool = True
    use_openai: bool = True

    # Llama.cpp
    llama_model_path: str = "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"
    # Context window requested from llama.cpp. The Qwen model may support larger
    # contexts, but this project defaults to 65,536 tokens to keep memory use sane.
    llama_ctx: int = 65536                  # Qwen = 65536/101000
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llama_n_gpu_layers: int = 20             # -1 offloads all layers when GPU backend is available
    llama_n_batch: int = 256                 # prompt processing batch
    llama_n_ubatch: int = 256                # physical micro-batch; None to let llama.cpp choose
    llama_low_vram: bool = True              # reduce Metal VRAM usage

    # MLX local model directory. Relative paths are resolved under ~/models.
    mlx_model_path: str = "Qwen2.5-7B-Instruct-1M-4bit"

    # External LLM (OpenAI-compatible endpoint). Used when USE_EXTERNAL_AI=true.
    llm_server_url: str = "http://127.0.0.1:8001/v1"
    llm_server_api_key: str = ""
    llm_server_model: str = "local-llm"
    external_base_url: str = "https://inference.do-ai.run/v1/chat/completions"
    external_model: str = "Qwen2.5-7B-Instruct-1M"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8001

    # RAG
    rag_top_k: int = 3
    rag_num_candidates: int = 50

    # NER service
    ner_url: str = "http://127.0.0.1:8000/ner"
    ner_timeout: float = 8.0

    # Internet search
    web_search_enabled: bool = True
    web_search_provider: str = "tavily"  # supported: tavily, duckduckgo
    web_search_fallback: str = "duckduckgo"
    web_search_max_results: int = 5
    web_search_timeout: float = 15.0
    tavily_api_key: str = ""
    duckduckgo_region: str = "wt-wt"

    # Remote Mem0 SDK connection (v2.2.1 self-hosted server)
    mem0_api_url: str = "http://127.0.0.1:8888"
    mem0_api_key: str = ""
    mem0_timeout: float = 30.0
    mem0_top_k: int = 5
    mem0_search_threshold: float = 0.1

    # Fixed workshop scopes
    mem0_user_id: str = "user-123"
    mem0_session_id: str = "123456"
    conversation_chunk_chars: int = 3000

    @property
    def mem0_complete_run_id(self) -> str:
        """Return the fixed run used for lossless conversation snapshots."""
        return f"{self.mem0_session_id}-complete"

    @property
    def mem0_synthesized_run_id(self) -> str:
        """Return the fixed run used for inferred conversational memories."""
        return f"{self.mem0_session_id}-synthesized"

    @property
    def mem0_internet_run_id(self) -> str:
        """Return the fixed run used for raw internet episodes."""
        return f"{self.mem0_session_id}-internet"


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable using common truthy strings."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_llama_n_gpu_layers() -> int:
    """Use full Metal offload by default on Apple Silicon."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return -1
    return Settings.llama_n_gpu_layers


def _is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon hardware."""

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _search_provider(name: str, default: str) -> str:
    """Normalize a configured internet-search provider."""
    value = os.getenv(name, default).strip().lower()
    if value not in ("tavily", "duckduckgo"):
        return default
    return value


def load_settings(env_file: str | None = None) -> Settings:
    """Load runtime settings from environment variables and an optional .env file."""
    # Load an explicit .env first; otherwise check the current working directory
    # and then the project root inferred from this file location.
    if env_file:
        load_dotenv(env_file)
    else:
        for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
            if candidate.exists():
                load_dotenv(str(candidate))
                break

    opensearch_port = _get_int("OPENSEARCH_PORT", Settings.opensearch_port)
    if opensearch_port != 9200:
        print("\n-----------------------------------------------------")
        print(f"WARNING: Using NON STANDARD OpenSearch Port: {opensearch_port}")
        print("Default OpenSearch Port is 9200")
        print("-----------------------------------------------------\n")

    use_openai = os.getenv("USE_OPENAI", "true").lower() in ("1", "true", "yes", "y", "on")
    external_ai = os.getenv("USE_EXTERNAL_AI", "true").lower() in ("1", "true", "yes", "y", "on")
    if use_openai:
        external_ai = True

    llm_server_url = os.getenv("LLM_SERVER_URL", Settings.llm_server_url)
    if external_ai:
        if use_openai:
            llm_server_url = "https://api.openai.com/v1"
        else:
            llm_server_url = _get_str("EXTERNAL_LLM_URL", Settings.external_base_url)

    llm_server_api_key = os.getenv("LLM_SERVER_API_KEY", Settings.llm_server_api_key)
    if external_ai:
        if use_openai:
            llm_server_api_key = _get_str("OPENAI_API_KEY", "")
        else:
            llm_server_api_key = _get_str("EXTERNAL_LLM_API_KEY", "")

    llm_server_model = os.getenv("LLM_SERVER_MODEL", Settings.llm_server_model)
    if external_ai:
        if use_openai:
            llm_server_model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        else:
            llm_server_model = os.getenv("EXTERNAL_LLM_MODEL", Settings.external_model)

    llama_ctx = _get_int("LLAMA_CTX", Settings.llama_ctx)
    if external_ai or use_openai:
        llama_ctx = _get_int("EXTERNAL_LLM_MAX_TOKENS", 127000)

    web_search_provider = (
        os.getenv("WEB_SEARCH_PROVIDER", Settings.web_search_provider).strip().lower()
    )
    if web_search_provider not in ("tavily", "duckduckgo"):
        web_search_provider = Settings.web_search_provider

    return Settings(
        opensearch_host=os.getenv("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", Settings.opensearch_index),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),

        # which LLM
        external_ai=external_ai,
        use_openai=use_openai,

        # LLaMA
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=llama_ctx,
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", _default_llama_n_gpu_layers()),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        llama_low_vram=_get_bool("LLAMA_LOW_VRAM", Settings.llama_low_vram),

        # MLX
        mlx_model_path=os.getenv(
            "MLX_MODEL_PATH",
            str(Path.home() / "models" / Settings.mlx_model_path),
        ),

        # External LLM (OpenAI-compatible endpoint)
        llm_server_url=llm_server_url,
        llm_server_api_key=llm_server_api_key,
        llm_server_model=llm_server_model,

        # Server
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),

        # RAG
        rag_top_k=_get_int("RAG_TOP_K", Settings.rag_top_k),
        rag_num_candidates=_get_int("RAG_NUM_CANDIDATES", Settings.rag_num_candidates),

        # NER service
        ner_url=os.getenv("NER_SERVER_URL", Settings.ner_url),
        ner_timeout=_get_float("NER_TIMEOUT", Settings.ner_timeout),

        # Internet search
        web_search_enabled=_get_bool("WEB_SEARCH_ENABLED", Settings.web_search_enabled),
        web_search_provider=web_search_provider,
        web_search_fallback=_search_provider("WEB_SEARCH_FALLBACK", Settings.web_search_fallback),
        web_search_max_results=_get_int(
            "WEB_SEARCH_MAX_RESULTS", Settings.web_search_max_results
        ),
        web_search_timeout=_get_float(
            "WEB_SEARCH_TIMEOUT", Settings.web_search_timeout
        ),
        tavily_api_key=os.getenv("TAVILY_API_KEY", Settings.tavily_api_key),
        duckduckgo_region=os.getenv("DUCKDUCKGO_REGION", Settings.duckduckgo_region),

        # Remote Mem0 SDK connection (v2.2.1 self-hosted server)
        mem0_api_url=_get_str("MEM0_API_URL", Settings.mem0_api_url),
        mem0_api_key=os.getenv("MEM0_API_KEY", Settings.mem0_api_key).strip(),
        mem0_timeout=_get_float("MEM0_TIMEOUT", Settings.mem0_timeout),
        mem0_top_k=_get_int("MEM0_TOP_K", Settings.mem0_top_k),
        mem0_search_threshold=_get_float(
            "MEM0_SEARCH_THRESHOLD", Settings.mem0_search_threshold
        ),
        mem0_user_id=_get_str("MEMORY_USER_ID", Settings.mem0_user_id),
        mem0_session_id=_get_str("MEMORY_SESSION_ID", Settings.mem0_session_id),
        conversation_chunk_chars=_get_int(
            "CONVERSATION_CHUNK_CHARS", Settings.conversation_chunk_chars
        ),
    )


__all__ = [
    "Settings",
    "load_settings",
    "_get_str",
    "_get_int",
    "_get_float",
    "_get_bool",
]
