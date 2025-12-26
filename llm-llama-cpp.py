
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os
os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"
# Pick a GGUF file from an HF repo (example below)
repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
fname = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

path = hf_hub_download(repo, filename=fname)
llm = Llama(model_path=path, n_threads=4, n_ctx=4096, verbose=False)


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

resp = llm.create_chat_completion(
    messages=messages, # type: ignore
    max_tokens=512,
    temperature=0.7,
    stream=True
)

for chunk in resp:
    content = chunk["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
print()  # Newline at the end

