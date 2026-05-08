"""
Shared configuration for model mappings used across plotting functions.
This module centralizes model size mappings, short names, provider groupings, and colors
to eliminate code duplication.
"""

# Model size mapping (in billions of parameters)
MODEL_SIZE_MAPPING = {
    "deepseek-r1:7b": 7,
    "codegemma:7b": 7,
    "mistral:latest": 7,  # Assuming Mistral-7B
    "qwen2:latest": 7,  # Assuming 7B
    "qwen2.5:latest": 7,  # Assuming 7B
    "phi3.5:3.8b-mini-instruct-fp16": 3.8,
    "codellama:13b": 13,
    "deepseek-r1:14b": 14,
    "phi4:latest": 14,
    "deepseek-coder-v2:16b": 16,
    "deepseek-r1:32b": 32,
    "qwen2.5-coder:32b": 32,
    "llama3.3:latest": 70,  # Assuming 70B
    "openai/4o": 128,  # Estimated size
    "o1-mini": 40,  # Estimated size
    "o1": 256,  # Estimated size
    "o3-mini": 128,  # Estimated size
    "o3-mini-low": 128,  # Estimated size
    "o3-mini-medium": 128,  # Estimated size
    "o3-mini-high": 128,  # Estimated size
    "o3-low-azure": 256,
    "o3-medium-azure": 256,
    "o3-high-azure": 256,
    "claude-3.5": 128,  # Estimated size
    "claude-3.7": 256,  # Estimated size
    "claude-3.7-thinking-8k": 256,  # Estimated size
    "claude-3.7-thinking-16k": 256,  # Estimated size
    "claude-3.7-thinking-16k-bedrock": 256,  # Estimated size
    "claude-3.7-thinking-32k-bedrock": 256,  # Estimated size
    "claude-3.7-thinking-64k-bedrock": 256,  # Estimated size
    "deepseek-r1-abacus": 256,
    "abacus-gemini-2-5-pro": 256,
    "top-5": 200,  # Special size for top-5
    "top-5-inverted-weighting": 200,
    "top-5-vendor": 200, 
    "top-5-parallel": 250,  # Highest score, larger size
}

# Model name shortening mapping
MODEL_SHORTNAMES = {
    "deepseek-r1:7b": "ds_r1_7b",
    "deepseek-r1:14b": "ds_r1_14b",
    "deepseek-r1:32b": "ds_r1_32b",
    "deepseek-coder-v2:16b": "ds_coder_16b",
    "codegemma:7b": "gemma_7b",
    "mistral:latest": "mistral",
    "qwen2:latest": "qwen2",
    "qwen2.5:latest": "qwen2.5",
    "qwen2.5-coder:32b": "qwen2.5_32b",
    "phi3.5:3.8b-mini-instruct-fp16": "phi3.5",
    "phi4:latest": "phi4",
    "codellama:13b": "codellama_13b",
    "llama3.3:latest": "llama3.3",
    "openai/4o": "4o",
    "o1-mini": "o1-mini",
    "o1": "o1",
    "o3-mini": "o3-mini",
    "o3-mini-low": "o3-mini-low",
    "o3-mini-medium": "o3-mini-medium",
    "o3-mini-high": "o3-mini-high",
    "o3-low-azure": "o3-low",
    "o3-medium-azure": "o3-medium",
    "o3-high-azure": "o3-high",
    "claude-3.5": "c3.5",
    "claude-3.7": "c3.7",
    "claude-3.7-thinking-8k": "c3.7-t8k",
    "claude-3.7-thinking-16k": "c3.7-t16k",
    "claude-3.7-thinking-16k-bedrock": "c3.7-t16k",
    "claude-3.7-thinking-32k-bedrock": "c3.7-t32k",
    "claude-3.7-thinking-64k-bedrock": "c3.7-t64k",
    "deepseek-r1-abacus": "ds_r1",
    "abacus-gemini-2-5-pro": "gemini-2.5-pro",
    "top-5": "top-5",
    "top-5-inverted-weighting": "top-5-inverted-weighting",
    "top-5-vendor": "top-5-vendor", 
    "top-5-parallel": "top-5-parallel",
}

# Model provider groupings
OPENAI_MODELS = [
    "openai/4o",
    "o1-mini",
    "o1",
    "o3-mini",
    "o3-mini-medium",
    "o3-mini-high",
    "o3-mini-low",
    "o3-low-azure",
    "o3-medium-azure",
    "o3-high-azure",
]

ANTHROPIC_MODELS = [
    "claude-3.5",
    "claude-3.7",
    "claude-3.7-thinking-8k",
    "claude-3.7-thinking-16k",
    "claude-3.7-thinking-16k-bedrock",
    "claude-3.7-thinking-32k-bedrock",
    "claude-3.7-thinking-64k-bedrock",
]

MISTRAL_MODELS = ["mistral:latest"]

META_MODELS = ["llama3.3:latest", "codellama:13b"]

DEEPSEEK_MODELS = [
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-coder-v2:16b",
    "deepseek-r1-abacus",
]

QWEN_MODELS = ["qwen2:latest", "qwen2.5:latest", "qwen2.5-coder:32b"]

MICROSOFT_MODELS = ["phi3.5:3.8b-mini-instruct-fp16", "phi4:latest"]

GOOGLE_MODELS = ["codegemma:7b", "abacus-gemini-2-5-pro"]

TOP5_MODELS = ["top-5", "top-5-inverted-weighting", "top-5-vendor", "top-5-parallel"]

# Full display names for publication figures
MODEL_FULLNAMES = {
    "deepseek-r1:7b": "DeepSeek R1 7B",
    "deepseek-r1:14b": "DeepSeek R1 14B",
    "deepseek-r1:32b": "DeepSeek R1 32B",
    "deepseek-coder-v2:16b": "DeepSeek Coder V2 16B",
    "deepseek-r1-abacus": "DeepSeek R1",
    "codegemma:7b": "CodeGemma 7B",
    "mistral:latest": "Mistral 7B",
    "qwen2:latest": "Qwen2 7B",
    "qwen2.5:latest": "Qwen2.5 7B",
    "qwen2.5-coder:32b": "Qwen2.5 Coder 32B",
    "phi3.5:3.8b-mini-instruct-fp16": "Phi 3.5 Mini 3.8B",
    "phi4:latest": "Phi 4 14B",
    "codellama:13b": "Code Llama 13B",
    "llama3.3:latest": "Llama 3.3 70B",
    "openai/4o": "OpenAI GPT-4o",
    "o1-mini": "OpenAI o1-mini",
    "o1": "OpenAI o1",
    "o3-mini": "OpenAI o3-mini",
    "o3-mini-low": "OpenAI o3-mini Low",
    "o3-mini-medium": "OpenAI o3-mini Medium",
    "o3-mini-high": "OpenAI o3-mini High",
    "o3-low-azure": "OpenAI o3 Low",
    "o3-medium-azure": "OpenAI o3 Medium",
    "o3-high-azure": "OpenAI o3 High",
    "claude-3.5": "Claude 3.5 Sonnet",
    "claude-3.7": "Claude 3.7 Sonnet",
    "claude-3.7-thinking-8k": "Claude 3.7 Sonnet Thinking 8k",
    "claude-3.7-thinking-16k": "Claude 3.7 Sonnet Thinking 16k",
    "claude-3.7-thinking-16k-bedrock": "Claude 3.7 Sonnet Thinking 16k",
    "claude-3.7-thinking-32k-bedrock": "Claude 3.7 Sonnet Thinking 32k",
    "claude-3.7-thinking-64k-bedrock": "Claude 3.7 Sonnet Thinking 64k",
    "abacus-gemini-2-5-pro": "Gemini 2.5 Pro",
    "top-5": "Top-5 Ensemble",
    "top-5-inverted-weighting": "Top-5 Inverted Weighting",
    "top-5-vendor": "Top-5 Vendor",
    "top-5-parallel": "Top-5 Parallel",
    "router": "Router",
}

# Reasoning models that should use square markers
REASONING_MODELS = [
    # OpenAI O-series models (reasoning)
    "o1-mini",
    "o1",
    "o3-mini",
    "o3-mini-low",
    "o3-mini-medium", 
    "o3-mini-high",
    "o3-low-azure",
    "o3-medium-azure",
    "o3-high-azure",
    # Claude thinking models (reasoning)
    "claude-3.7-thinking-8k",
    "claude-3.7-thinking-16k",
    "claude-3.7-thinking-16k-bedrock",
    "claude-3.7-thinking-32k-bedrock",
    "claude-3.7-thinking-64k-bedrock",
    # Other reasoning models
    "abacus-gemini-2-5-pro",
    "deepseek-coder-v2:16b", 
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-r1-abacus",
]

# Provider color mapping
PROVIDER_COLORS = {
    "openai": "olive",
    "anthropic": "brown",
    "mistral": "red",
    "meta": "darkblue",
    "deepseek": "gray",
    "qwen": "purple",
    "microsoft": "orange",
    "google": "skyblue",
    "top5": "gold",  # Special color for top-5
    "unknown": "black",
}

# Font size configuration for consistent plotting
# Based on create_scatter_plot_only in utils.py
AXIS_LABEL_FONTSIZE = 18  # Font size for x-axis and y-axis labels
TICK_LABEL_FONTSIZE = 17  # Font size for x-axis and y-axis tick labels

def get_model_provider(model_name):
    """
    Get the provider for a given model name.
    
    Parameters:
    -----------
    model_name : str
        The full model name
        
    Returns:
    --------
    str
        The provider name
    """
    if model_name in TOP5_MODELS:
        return "top5"
    elif model_name in OPENAI_MODELS:
        return "openai"
    elif model_name in ANTHROPIC_MODELS:
        return "anthropic"
    elif model_name in MISTRAL_MODELS:
        return "mistral"
    elif model_name in META_MODELS:
        return "meta"
    elif model_name in DEEPSEEK_MODELS:
        return "deepseek"
    elif model_name in QWEN_MODELS:
        return "qwen"
    elif model_name in MICROSOFT_MODELS:
        return "microsoft"
    elif model_name in GOOGLE_MODELS:
        return "google"
    else:
        return "unknown"

def get_model_color(model_name):
    """
    Get the color for a given model name.
    
    Parameters:
    -----------
    model_name : str
        The full model name
        
    Returns:
    --------
    str
        The color name
    """
    # Special handling for specific top-5 models with different shades of yellow
    # if model_name == "top-5":
    #     return "gold"
    # elif model_name == "top-5-parallel":
    #     return "darkgoldenrod"
    # elif model_name == "top-5-inverted-weighting":
    #     return "goldenrod"
    # elif model_name == "top-5-vendor":
    #     return "orange"

    if model_name in TOP5_MODELS:
        return "gold"
    
    provider = get_model_provider(model_name)
    return PROVIDER_COLORS[provider]

def get_model_size(model_name, default=10):
    """
    Get the size (in billions of parameters) for a given model name.
    
    Parameters:
    -----------
    model_name : str
        The full model name
    default : float
        Default size if model not found
        
    Returns:
    --------
    float
        The model size in billions of parameters
    """
    return MODEL_SIZE_MAPPING.get(model_name, default)

def get_model_shortname(model_name):
    """
    Get the shortened name for a given model name.
    
    Parameters:
    -----------
    model_name : str
        The full model name
        
    Returns:
    --------
    str
        The shortened model name
    """
    return MODEL_SHORTNAMES.get(model_name, model_name)

def get_model_fullname(model_name):
    """
    Get the full display name for a given model name.

    Parameters:
    -----------
    model_name : str
        The internal model name

    Returns:
    --------
    str
        The full display name
    """
    return MODEL_FULLNAMES.get(model_name, model_name)

def get_model_marker(model_name):
    """
    Get the marker type for a given model name.
    
    Parameters:
    -----------
    model_name : str
        The full model name
        
    Returns:
    --------
    str
        The marker type ('*' for top-5 models, 's' for reasoning models, 'o' for others)
    """
    if model_name in TOP5_MODELS:
        return "*"
    elif model_name in REASONING_MODELS:
        return "s"  # Square marker for reasoning models
    else:
        return "o"  # Circle marker for non-reasoning models
