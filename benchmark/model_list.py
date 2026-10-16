meta_models = [
    # "top-5",
    # "top-5-inverted-weighting",
    # "top-5-vendor",
    # "top-5-parallel"
]

# Models that return thinking/reasoning in their response
THINKING_MODELS = [
    # O3 and GPT-5 models (also return reasoning_tokens)
    "o3-low-azure",
    "o3-medium-azure",
    "o3-high-azure",
    "gpt-5-high",
    "gpt-5-low",
    "gpt-5.6-sol",
    "claude-opus-5",
    # Claude thinking models
    "claude-3.7-thinking-8k",
    "claude-3.7-thinking-16k",
    "claude-3.7-thinking-8k-bedrock",
    "claude-3.7-thinking-16k-bedrock",
    "claude-3.7-thinking-32k-bedrock",
    "claude-3.7-thinking-64k-bedrock",
    # DeepSeek models
    "deepseek-r1-abacus",
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-coder-v2:16b",
]


# Uncomment the models to run. These are the 32 individual configurations
# evaluated in the paper.
models = [
    # OpenAI
    # "openai/4o",
    # "o1-mini",
    # "o1",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o3-low-azure",
    # "o3-medium-azure",
    # "o3-high-azure",
    # "gpt-5.6-sol",
    # Anthropic
    # "claude-3.5",
    # "claude-3.7",
    # "claude-3.7-thinking-8k",
    # "claude-3.7-thinking-16k-bedrock",
    # "claude-3.7-thinking-32k-bedrock",
    # "claude-3.7-thinking-64k-bedrock",
    # "claude-opus-5",
    # Google (stored in results as "abacus-gemini-2-5-pro")
    # "gemini-2-5-pro-abacus",
    # DeepSeek
    # "deepseek-r1-abacus",
    # "deepseek-r1:7b",
    # "deepseek-r1:14b",
    # "deepseek-r1:32b",
    # "deepseek-coder-v2:16b",
    # Local models via Ollama
    # "phi3.5:3.8b-mini-instruct-fp16",
    # "phi4:latest",
    # "llama3.3:latest",
    # "qwen2.5-coder:32b",
    # "qwen2.5:latest",
    # "qwen2:latest",
    # "codegemma:7b",
    # "codellama:13b",
    # "mistral:latest",
]
