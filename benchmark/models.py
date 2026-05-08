import os
from functools import lru_cache
from retry import retry

from openai import AzureOpenAI
from ollama_utils import execute_llm_call
import abacusai
from llm_manager import LLMManager

# Initialize the manager
manager = LLMManager()


# AWS Bedrock support (optional dependency)
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # keep the module import-safe on systems w/o boto3
    boto3, ClientError = None, None


def chat_with_model(
    prompt: str, model: str, max_tokens: int = 4000, temperature: float = 0
) -> str:
    if model == "openai/4o":
        client = AzureOpenAI(
            api_key=os.environ.get("AZURE_API_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.environ.get("AZURE_API_BASE"),
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT"),
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content

    elif model == "o1":
        response = manager.generate(
            prompt,
            model="azure/o1",
        )

        return response.choices[0].message.content

    elif model == "openai/o1-mini" or model == "o1-mini":
        response = manager.generate(
            prompt,
            model="azure/o1-mini",
        )

        return response.choices[0].message.content

    elif model == "o3-mini-low":
        client = AzureOpenAI(
            api_key=os.environ.get("AZURE_o1_API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.environ.get("AZURE_o1_API_BASE"),
            azure_deployment=os.environ.get("AZURE_o3_mini_DEPLOYMENT"),
        )

        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
        )

        return response.choices[0].message.content

    elif model == "o3-mini-medium":
        client = AzureOpenAI(
            api_key=os.environ.get("AZURE_o1_API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.environ.get("AZURE_o1_API_BASE"),
            azure_deployment=os.environ.get("AZURE_o3_mini_DEPLOYMENT"),
        )

        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="medium",
        )

        return response.choices[0].message.content

    elif model == "o3-mini-high":
        client = AzureOpenAI(
            api_key=os.environ.get("AZURE_o1_API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.environ.get("AZURE_o1_API_BASE"),
            azure_deployment=os.environ.get("AZURE_o3_mini_DEPLOYMENT"),
        )

        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
        )

        return response.choices[0].message.content

    elif model == "o3-low-azure":
        response = manager.generate(
            prompt,
            model="azure/o3",
            reasoning_effort="low",
            summary_level="detailed",
        )
        return response

    elif model == "o3-medium-azure":
        response = manager.generate(
            prompt,
            model="azure/o3",
            reasoning_effort="medium",
            summary_level="detailed",
        )

        return response

    elif model == "o3-high-azure":
        response = manager.generate(
            prompt,
            model="azure/o3",
            reasoning_effort="high",
            summary_level="detailed",
        )

        return {
            "thinking": response.output[0].summary[0].text if response.output[0].summary else "",
            "response": response.output[1].content[0].text,
            "reasoning_tokens": getattr(response.usage.output_tokens_details, 'reasoning_tokens', None)
        }

    elif model == "gpt-5-low":
        response = manager.generate(
            prompt,
            model="azure/gpt-5",
            reasoning_effort="low",
            summary_level="detailed",
        )

        return response

    elif model == "gpt-5-high":
        response = manager.generate(
            prompt,
            model="azure/gpt-5",
            reasoning_effort="high",
            summary_level="detailed",
        )

        return response

    elif model == "claude-3.7-bedrock":
        response = manager.generate(
            prompt,
            model="bedrock/claude-sonnet-3.7",
            temperature=0.7
        )

        return response

    elif model == "claude-3.7-thinking-8k-bedrock":
        response = manager.generate(
            prompt,
            model="bedrock/claude-sonnet-3.7",
            max_tokens=20000,
            thinking_tokens=8000,
            temperature=1,
        )

        return response

    elif model == "claude-3.7-thinking-16k-bedrock":
        response = manager.generate(
            prompt,
            model="bedrock/claude-sonnet-3.7",
            max_tokens=32000,
            thinking_tokens=16000,
            temperature=1,
        )

        return response

    elif model == "claude-3.7-thinking-32k-bedrock":
        response = manager.generate(
            prompt,
            model="bedrock/claude-sonnet-3.7",
            max_tokens=64000,
            thinking_tokens=32000,
            temperature=1,
        )

        return response

    elif model == "claude-3.7-thinking-64k-bedrock":
        response = manager.generate(
            prompt,
            model="bedrock/claude-sonnet-3.7",
            max_tokens=128000,
            thinking_tokens=64000,
            temperature=1,
        )

        return response

    elif model == "gemini-2-5-pro-abacus":
        response = manager.generate(
            prompt,
            model_name="abacus/gemini-2-5-pro-abacus"
        )

        return response.content

    elif model == "deepseek-r1-abacus":
        response = manager.generate(
            prompt,
            model_name="abacus/deepseek-r1"
        )

        return response.content

    elif model == "gemini-2-flash-thinking-abacus":
        api_key = os.getenv("ABACUS_API_KEY")
        llm = abacusai.ApiClient(api_key=api_key)

        response = llm.evaluate_prompt(
            prompt=prompt,
            system_message="",
            llm_name="DEEPSEEK_R1",
            temperature=0.7,
        )

        return response.content

    else:
        return execute_llm_call(model, prompt, temperature)


@lru_cache(maxsize=10000)
@retry(tries=3)
def embed(text: str) -> list[float]:

    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_API_KEY_OLD"),
        api_version="2023-05-15",
        azure_endpoint=os.environ.get("AZURE_API_BASE_OLD"),
        azure_deployment=os.environ.get(
            "AZURE_EMBEDDING_DEPLOYMENT"
        ),  # e.g., "https://your-resource.openai.azure.com/"
    )
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large",  # or your deployed embedding model name
        # your Azure deployment name
    )
    # Extract the embedding from the response
    embedding = response.data[0].embedding
    return embedding
