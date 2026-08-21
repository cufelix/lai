"""Every OpenAI-compatible backend LAI knows how to reach.

Almost the whole industry now speaks the OpenAI chat-completions dialect, so
supporting one more vendor is data, not code: a base URL, the environment
variable people keep their key in, and a sane default model. Keeping that as a
table means `lai models` can list what a machine could use without anybody
hard-coding vendor names into the agent.

Local runtimes are in here too (LM Studio, llama.cpp, vLLM, LiteLLM). They need
no key at all, which makes them the honest answer to "can I run this without
paying anyone" — and they are checked by opening a socket, not by guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Vendor:
    """One OpenAI-compatible endpoint."""

    name: str
    label: str
    base_url: str
    default_model: str
    env_keys: tuple[str, ...] = ()
    """Environment variables that would hold the key, best first."""
    signup: str = ""
    local: bool = False
    """True for something running on this machine, where no key is needed."""
    vision: bool = True
    notes: str = ""
    model_env: str = ""
    """Environment variable overriding the default model."""
    url_env: str = ""
    """Environment variable holding the base URL, when it is per-account.

    Azure OpenAI gives every resource its own hostname, and a self-hosted
    endpoint behind a company domain is the same shape. Those cannot be a
    constant in a table, but they are still just data — one more variable.
    """
    extra_headers: dict = field(default_factory=dict)

    @property
    def needs_key(self) -> bool:
        return not self.local and bool(self.env_keys)

    def url(self, env=None) -> str:
        """The base URL, taking the per-account one when this vendor has one."""
        import os  # noqa: PLC0415

        env = os.environ if env is None else env
        if self.url_env:
            configured = str(env.get(self.url_env, "")).strip().rstrip("/")
            if configured:
                return configured
        return self.base_url


# Hosted vendors, roughly in order of how likely someone is to have one.
VENDORS: tuple[Vendor, ...] = (
    Vendor("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4o",
           ("OPENAI_API_KEY",), "https://platform.openai.com/api-keys",
           model_env="OPENAI_MODEL"),
    Vendor("openrouter", "OpenRouter — one key, most models",
           "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.5",
           ("OPENROUTER_API_KEY",), "https://openrouter.ai/keys",
           model_env="OPENROUTER_MODEL"),
    Vendor("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
           "gemini-2.0-flash", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
           "https://aistudio.google.com/apikey", model_env="GEMINI_MODEL"),
    Vendor("groq", "Groq — very fast inference", "https://api.groq.com/openai/v1",
           "llama-3.3-70b-versatile", ("GROQ_API_KEY",), "https://console.groq.com/keys",
           vision=False, model_env="GROQ_MODEL"),
    Vendor("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat",
           ("DEEPSEEK_API_KEY",), "https://platform.deepseek.com/api_keys",
           vision=False, model_env="DEEPSEEK_MODEL"),
    Vendor("mistral", "Mistral", "https://api.mistral.ai/v1", "mistral-large-latest",
           ("MISTRAL_API_KEY",), "https://console.mistral.ai/api-keys",
           model_env="MISTRAL_MODEL"),
    Vendor("xai", "xAI Grok", "https://api.x.ai/v1", "grok-2-vision-1212",
           ("XAI_API_KEY", "GROK_API_KEY"), "https://console.x.ai",
           model_env="XAI_MODEL"),
    Vendor("together", "Together AI", "https://api.together.xyz/v1",
           "meta-llama/Llama-3.3-70B-Instruct-Turbo", ("TOGETHER_API_KEY",),
           "https://api.together.ai/settings/api-keys", vision=False,
           model_env="TOGETHER_MODEL"),
    Vendor("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1",
           "accounts/fireworks/models/llama-v3p3-70b-instruct", ("FIREWORKS_API_KEY",),
           "https://fireworks.ai/account/api-keys", vision=False,
           model_env="FIREWORKS_MODEL"),
    Vendor("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "llama-3.3-70b",
           ("CEREBRAS_API_KEY",), "https://cloud.cerebras.ai", vision=False,
           model_env="CEREBRAS_MODEL"),
    Vendor("perplexity", "Perplexity", "https://api.perplexity.ai", "sonar-pro",
           ("PERPLEXITY_API_KEY", "PPLX_API_KEY"), "https://www.perplexity.ai/settings/api",
           vision=False, model_env="PERPLEXITY_MODEL"),
    Vendor("nebius", "Nebius AI Studio", "https://api.studio.nebius.ai/v1",
           "Qwen/Qwen2.5-72B-Instruct", ("NEBIUS_API_KEY",), "https://studio.nebius.ai",
           vision=False, model_env="NEBIUS_MODEL"),
    Vendor("moonshot", "Moonshot / Kimi", "https://api.moonshot.cn/v1", "moonshot-v1-32k",
           ("MOONSHOT_API_KEY",), "https://platform.moonshot.cn/console/api-keys",
           vision=False, model_env="MOONSHOT_MODEL"),
    Vendor("qwen", "Alibaba Qwen (DashScope)",
           "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max",
           ("DASHSCOPE_API_KEY", "QWEN_API_KEY"), "https://dashscope.console.aliyun.com",
           model_env="QWEN_MODEL"),
    Vendor("zai-coding", "Z.ai coding plan (OpenAI dialect)",
           "https://api.z.ai/api/coding/paas/v4", "glm-4.6",
           ("ZAI_API_KEY", "GLM_API_KEY"), "https://z.ai",
           model_env="ZAI_MODEL", notes="the coding-plan endpoint, priced per plan not per token"),
    Vendor("sambanova", "SambaNova", "https://api.sambanova.ai/v1",
           "Meta-Llama-3.3-70B-Instruct", ("SAMBANOVA_API_KEY",),
           "https://cloud.sambanova.ai/apis", vision=False, model_env="SAMBANOVA_MODEL"),
    Vendor("hyperbolic", "Hyperbolic", "https://api.hyperbolic.xyz/v1",
           "meta-llama/Llama-3.3-70B-Instruct", ("HYPERBOLIC_API_KEY",),
           "https://app.hyperbolic.xyz/settings", model_env="HYPERBOLIC_MODEL"),
    Vendor("novita", "Novita AI", "https://api.novita.ai/v3/openai",
           "meta-llama/llama-3.3-70b-instruct", ("NOVITA_API_KEY",),
           "https://novita.ai/settings/key-management", model_env="NOVITA_MODEL"),
    Vendor("github", "GitHub Models", "https://models.github.ai/inference",
           "openai/gpt-4o", ("GITHUB_TOKEN", "GITHUB_MODELS_TOKEN"),
           "https://github.com/settings/tokens", model_env="GITHUB_MODEL",
           notes="free tier for anyone with a GitHub token"),
    Vendor("huggingface", "Hugging Face Inference",
           "https://router.huggingface.co/v1", "meta-llama/Llama-3.3-70B-Instruct",
           ("HF_TOKEN", "HUGGINGFACE_API_KEY"), "https://huggingface.co/settings/tokens",
           model_env="HF_MODEL"),
    Vendor("azure", "Azure OpenAI", "https://YOUR-RESOURCE.openai.azure.com/openai/v1",
           "gpt-4o", ("AZURE_OPENAI_API_KEY",), "https://portal.azure.com",
           model_env="AZURE_OPENAI_MODEL", url_env="AZURE_OPENAI_ENDPOINT",
           notes="every resource has its own hostname — set AZURE_OPENAI_ENDPOINT"),
    Vendor("venice", "Venice AI (private)", "https://api.venice.ai/api/v1",
           "llama-3.3-70b", ("VENICE_API_KEY",), "https://venice.ai/settings/api",
           model_env="VENICE_MODEL", notes="no logging, no retention"),
    Vendor("chutes", "Chutes", "https://llm.chutes.ai/v1",
           "deepseek-ai/DeepSeek-V3", ("CHUTES_API_KEY",), "https://chutes.ai",
           vision=False, model_env="CHUTES_MODEL"),
    Vendor("baseten", "Baseten", "https://inference.baseten.co/v1",
           "deepseek-ai/DeepSeek-V3", ("BASETEN_API_KEY",), "https://app.baseten.co/settings/api_keys",
           vision=False, model_env="BASETEN_MODEL"),
    Vendor("lambda", "Lambda Inference", "https://api.lambda.ai/v1",
           "llama3.3-70b-instruct-fp8", ("LAMBDA_API_KEY",), "https://cloud.lambda.ai/api-keys",
           vision=False, model_env="LAMBDA_MODEL"),
    Vendor("inception", "Inception (diffusion LLMs)", "https://api.inceptionlabs.ai/v1",
           "mercury-coder", ("INCEPTION_API_KEY",), "https://platform.inceptionlabs.ai",
           vision=False, model_env="INCEPTION_MODEL"),
)

# Things running on this machine. Probed by connecting, never by a key.
LOCAL_VENDORS: tuple[Vendor, ...] = (
    Vendor("ollama", "Ollama", "http://127.0.0.1:11434/v1", "qwen3-vl:2b",
           (), "https://ollama.com", local=True, model_env="OLLAMA_MODEL",
           notes="the usual local choice; use a -vl model to keep vision"),
    Vendor("lmstudio", "LM Studio", "http://127.0.0.1:1234/v1", "local-model",
           (), "https://lmstudio.ai", local=True, model_env="LMSTUDIO_MODEL",
           notes="start the local server from LM Studio's Developer tab"),
    Vendor("llamacpp", "llama.cpp server", "http://127.0.0.1:8080/v1", "local-model",
           (), "https://github.com/ggml-org/llama.cpp", local=True,
           model_env="LLAMACPP_MODEL", notes="llama-server --port 8080"),
    Vendor("vllm", "vLLM", "http://127.0.0.1:8000/v1", "local-model",
           (), "https://docs.vllm.ai", local=True, model_env="VLLM_MODEL",
           notes="vllm serve <model>"),
    Vendor("litellm", "LiteLLM proxy", "http://127.0.0.1:4000/v1", "gpt-4o",
           ("LITELLM_API_KEY",), "https://docs.litellm.ai", local=True,
           model_env="LITELLM_MODEL", notes="one endpoint in front of many vendors"),
    Vendor("jan", "Jan", "http://127.0.0.1:1337/v1", "local-model",
           (), "https://jan.ai", local=True, model_env="JAN_MODEL"),
    Vendor("koboldcpp", "KoboldCpp", "http://127.0.0.1:5001/v1", "local-model",
           (), "https://github.com/LostRuins/koboldcpp", local=True,
           model_env="KOBOLD_MODEL"),
    Vendor("textgen", "text-generation-webui", "http://127.0.0.1:5000/v1", "local-model",
           (), "https://github.com/oobabooga/text-generation-webui", local=True,
           model_env="TEXTGEN_MODEL", notes="enable the OpenAI extension"),
    Vendor("localai", "LocalAI", "http://127.0.0.1:8080/v1", "local-model",
           (), "https://localai.io", local=True, model_env="LOCALAI_MODEL"),
)

ALL_VENDORS: tuple[Vendor, ...] = VENDORS + LOCAL_VENDORS
BY_NAME: dict[str, Vendor] = {vendor.name: vendor for vendor in ALL_VENDORS}


def get(name: str) -> Vendor | None:
    return BY_NAME.get((name or "").strip().lower())


def env_key_for(vendor: Vendor, env: dict) -> str:
    """The key this vendor would use from the given environment, if any."""
    for variable in vendor.env_keys:
        value = (env.get(variable) or "").strip()
        if value:
            return value
    return ""


def source_of(vendor: Vendor, env: dict) -> str:
    for variable in vendor.env_keys:
        if (env.get(variable) or "").strip():
            return variable
    return "local endpoint" if vendor.local else ""


__all__ = [
    "ALL_VENDORS",
    "BY_NAME",
    "LOCAL_VENDORS",
    "VENDORS",
    "Vendor",
    "env_key_for",
    "get",
    "source_of",
]
