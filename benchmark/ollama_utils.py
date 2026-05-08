from enum import Enum
from langchain.llms import Ollama
import re
import ollama

class LLMProviders(Enum):
    OLLAMA = "ollama"

base_models_path = {
    'mistral': ["mistral", LLMProviders.OLLAMA],
    'qwen2.5-coder': ["qwen2.5-coder:32b", LLMProviders.OLLAMA],
    'qwen2': ["qwen2", LLMProviders.OLLAMA],
    'qwen2.5': ["qwen2.5", LLMProviders.OLLAMA],
    'mistral-nemo': ["mistral-nemo", LLMProviders.OLLAMA],
    'phi3.5': ["phi3.5", LLMProviders.OLLAMA],
    'phi3.5-fp16': ["phi3.5:3.8b-mini-instruct-fp16", LLMProviders.OLLAMA],
    'athene-v2': ["athene-v2", LLMProviders.OLLAMA],
    'athene-v2-agent': ["hf.co/lmstudio-community/Athene-V2-Agent-GGUF", LLMProviders.OLLAMA],
    'llama3.3': ["llama3.3", LLMProviders.OLLAMA],
}

def load_ollama_model(base_model):
    """
    Load a text model with caching
    """
    base_model_path, provider = base_models_path[base_model]

    match provider:
        case LLMProviders.OLLAMA:
            llm = Ollama(base_url="http://localhost:11434", model=base_model_path, temperature=0)

    print(f"Loaded model '{base_model}'")

    return llm


def execute_llm_call(model, prompt, temperature):
    """
    Executes an LLM call with standardized provider handling.
    
    Args:
        llm: The loaded language model
        prompt_template: The formatted prompt template
        user_prompt: The user's input text
        strip_markdown: Whether to strip markdown formatting from output
        
    Returns:
        str: The processed LLM output
    """
    print("in Ollama")
    llm = Ollama(
    model=model,  # or any other model you have pulled
    temperature=temperature
)
    response = llm(prompt)
    
    return response


def display_ollama_models_list():
    # ollama_client = Ollama()
    # model_list = ollama_client.list_models()
    # for model in model_list: print(f"Model Name: {model.name}, Version: {model.version}, Description: {model.description}")

    for idx, m in enumerate(ollama.list()['models'], 3): print(f"{idx}. {m['model']}")
