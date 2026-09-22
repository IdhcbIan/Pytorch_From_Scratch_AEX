"""
█▀▄▀█ █▀█ █▀▄ ▄▀█ █░░   █░█░█ █░█ █ █▀ █▀█ █▀▀ █▀█
█░▀░█ █▄█ █▄▀ █▀█ █▄▄   ▀▄▀▄▀ █▀█ █ ▄█ █▀▀ ██▄ █▀▄

Aula: Deploy de API de Transcrição de Áudio com Whisper e Modal
Objetivo: Criar uma API compatível com a OpenAI rodando em GPU Serverless.
"""

import modal

# ==========================================
# 1. DEFINIÇÃO DO AMBIENTE E DEPENDÊNCIAS
# ==========================================
# Criamos a imagem do contêiner a partir de uma base otimizada da NVIDIA,
# adicionamos Python 3.12 e instalamos as bibliotecas necessárias.

whisper_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "transformers==4.48.0",
        "torch==2.5.1",
        "accelerate==1.2.1",
        "librosa==0.10.2",
        "fastapi[standard]==0.115.6",
        "python-multipart==0.0.20", # Necessário para o FastAPI receber arquivos
    )
)

# ==========================================
# 2. VOLUMES (CACHE PERSISTENTE)
# ==========================================
# O modelo Whisper pesa cerca de 1.6GB. Usamos um volume (como um "pen drive" na nuvem) 
# para guardar o modelo do Hugging Face. Assim, não precisamos baixar 1.6GB toda vez 
# que o contêiner ligar, poupando tempo e dinheiro.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# ==========================================
# 3. INICIALIZAÇÃO DO APP MODAL
# ==========================================
app = modal.App("whisper-inference")
MINUTES = 60 # Constante para facilitar a leitura do tempo (em segundos)

# ==========================================
# 4. CONFIGURAÇÃO DA FUNÇÃO SERVERLESS E GPU
# ==========================================
# Vinculamos a infraestrutura: Imagem, GPU T4, Tempo de Ociosidade e o Volume.
@app.function(
    image=whisper_image,
    gpu="T4",
    scaledown_window=15 * MINUTES,  # Mantém a GPU ligada por 15 min sem uso (evita cold start)
    timeout=30 * MINUTES,           # Tempo máximo de processamento por requisição
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)

@modal.asgi_app()
def serve():
    """
    FastAPI server for Whisper transcription inference.
    """
    # As importações pesadas ficam DENTRO da função para não quebrarem o ambiente local.
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.responses import JSONResponse
    from typing import Optional
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import librosa
    import tempfile
    import os

    # Inicializa o FastAPI
    web_app = FastAPI(
        title="Whisper Transcription API",
        description="Audio transcription API using Whisper Large V3 Turbo",
        version="1.0.0",
    )

    # ==========================================
    # 5. CARREGAMENTO DO MODELO (WARM-UP)
    # ==========================================
    # Isso roda assim que o contêiner liga, ANTES da primeira requisição,
    # deixando o modelo pronto na memória da GPU.
    MODEL_NAME = "openai/whisper-large-v3-turbo"
    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Cria o pipeline de inferência
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print(f"Whisper model loaded on {device}!")

    # ==========================================
    # 6. ROTAS DA API
    # ==========================================
    @web_app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "device": device,
        }

    @web_app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        task: str = Form("transcribe"),  # transcribe or translate
        temperature: float = Form(0.0),
    ):
        """Endpoint compatível com a OpenAI para transcrição de áudio."""
        try:
            # 1. Lê os bytes do arquivo enviado
            audio_bytes = await file.read()

            # 2. Salva em um arquivo temporário no disco do contêiner
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            try:
                # 3. Librosa carrega o áudio convertendo estritamente para 16kHz mono (exigência do Whisper)
                audio_data, _ = librosa.load(temp_path, sr=16000, mono=True)

                # 4. Configura os parâmetros de geração e faz a inferência
                generate_kwargs = {
                    "task": task,
                    "do_sample": temperature > 0,
                    "temperature": temperature if temperature > 0 else None,
                }

                if language:
                    generate_kwargs["language"] = language

                result = pipe(
                    audio_data,
                    return_timestamps=False,
                    generate_kwargs=generate_kwargs,
                )

                transcription = result["text"] if isinstance(result, dict) else str(result)

                # 5. Retorna o JSON no mesmo formato da OpenAI
                return JSONResponse(content={
                    "text": transcription.strip(),
                    "language": language if language else "auto",
                    "task": task,
                })

            finally:
                # Limpa o arquivo temporário
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app