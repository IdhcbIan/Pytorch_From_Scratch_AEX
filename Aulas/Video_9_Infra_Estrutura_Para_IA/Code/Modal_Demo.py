"""
--------------------------------

███╗░░░███╗░█████╗░██████╗░░█████╗░██╗░░░░░
████╗░████║██╔══██╗██╔══██╗██╔══██╗██║░░░░░
██╔████╔██║██║░░██║██║░░██║███████║██║░░░░░
██║╚██╔╝██║██║░░██║██║░░██║██╔══██║██║░░░░░
██║░╚═╝░██║╚█████╔╝██████╔╝██║░░██║███████╗
╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚═╝░░╚═╝╚══════╝

--------------------------------

Vamos usar o site de compute chamado Modal.com

https://modal.com/

-> Para rodar:

    - pip3 install modal
    - modal setup ( e fazer login no browser)
    - modal run Modal_Demo.py

--------------------------------
"""

import modal

# =============================================================================
# STEP 1: Define the container image
# =============================================================================
# Modal roda o codigo em containers, como se fossem pequenos ambientes onde temos todas as dependencias
#     assim criando uma bolha onde o codigo fica isolado, diferentemente de ambientes que temos acesso direto a um servidor

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "vllm==0.8.5",
        "huggingface_hub[hf_xet]>=0.30",
        "transformers>=4.51.0",
    )
)


# =============================================================================
# STEP 2: Configure the model
# =============================================================================
# O modelo Qwen3-8B, um bom modelo LLM open source da empresa chinesa Alibaba


MODEL_NAME = "Qwen/Qwen3-8B"


# =============================================================================
# STEP 3: Create persistent storage for model weights
# =============================================================================
# Criando um volume em disco para armazenar os pesos do modelo para que seja presciso 
#         baixar todas as veses que rodamos o codigo


hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# =============================================================================
# STEP 4: Create the Modal App
# =============================================================================


app = modal.App("qwen3-demo")

MINUTES = 60


# =============================================================================
# STEP 5: Define the GPU inference function
# =============================================================================
# Configutando a GPU!! Aqui usamos uma GPU H100 com 80GB de VRAM!!

@app.function(
    image=vllm_image,
    gpu="H100",
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
def generate(prompt: str, max_tokens: int = 512) -> str:
    """
    Generate text using Qwen3 on a cloud GPU.
    """
    from vllm import LLM, SamplingParams

    print(f"Loading model: {MODEL_NAME}")

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=8192,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
    )

    outputs = llm.generate([prompt], sampling_params)

    return outputs[0].outputs[0].text


# =============================================================================
# STEP 6: Entrypoint (runs locally, calls Modal for inference)
# =============================================================================

PROMPT = "Escreva um poema curto sobre como voce se sente rodando dentro de uma GPU."


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Qwen3-8B rodando em GPU na nuvem via Modal")
    print("=" * 60)
    print(f"\nPrompt: {PROMPT}\n")
    print("-" * 60)

    response = generate.remote(PROMPT)

    print(f"\nResposta:\n{response}")
    print("=" * 60)
