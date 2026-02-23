# fiap-ai-tech-challenge-04

Análise multimodal de vídeos de consulta (saúde da mulher): **análise de vídeo** (YOLOv8-pose + heurísticas de linguagem corporal) e **processamento de gravações de voz** (transcrição com faster-whisper e análise do texto). Executável em computador pessoal, sem GPU e sem serviços em nuvem.

## Requisitos

- Python 3.12
- Poetry

## Instalação

```bash
poetry install
```

Na primeira execução, o YOLOv8 (modelo `yolov8n-pose.pt`) e o faster-whisper (modelo `tiny`) serão baixados automaticamente. Depois disso, o fluxo roda **offline**.

## Uso

Coloque um vídeo de consulta (ex.: `data/demo_consultation.mp4`). Dois subcomandos:

**Análise (relatórios JSON):**

```bash
poetry run python main.py run data/demo_consultation.mp4
```

Opções: `--output-dir DIR`, `--skip-audio`, `--skip-video`.

**Vídeo anotado (caixas e rótulos no vídeo, com áudio e tempo correto):**

```bash
poetry run python main.py annotate-video data/demo_consultation.mp4
```

O vídeo anotado é gerado em `output/video_annotated_<nome>.mp4`. O áudio do vídeo original e os metadados de duração (tempo em segundos no player) são incluídos **em Python** via MoviePy; o texto com acentos (ex.: "Mãos no rosto") é desenhado com Pillow. O usuário não precisa executar ffmpeg manualmente.

Os relatórios (JSON) são gerados em `output/`:

- `video_report_<nome>.json` — Análise de pose e indicadores de desconforto/medo/defensivo.
- `audio_report_<nome>.json` — Transcrição e análise do texto (indicadores de ansiedade, hesitação, etc.).
- `consolidated_<nome>.json` — Resumo dos dois pipelines.

## Vídeo para demonstração

Sugestão: use um vídeo curto (30 s a 2 min) de cena de consulta (médico + paciente), com áudio de fala. Exemplos de fontes gratuitas:

- **Pexels:** "Doctor Talking to a Patient Sitting on Bed", "Women on Checkup in Doctors Office".
- **Mixkit:** busque por "female doctor consultation" ou "patient doctor conversation".

Coloque o arquivo em `data/` (ex.: `data/demo_consultation.mp4`) e rode:

```bash
poetry run python main.py run data/demo_consultation.mp4
```

## Fluxo multimodal

1. **Vídeo:** OpenCV lê o vídeo → YOLOv8-pose (modelo nano ou small, CPU) extrai keypoints → heurísticas detectam postura de desconforto, cabeça baixa, mãos no rosto, braços defensivos → relatório com segmentos e contagens.
2. **Áudio:** MoviePy extrai o áudio em WAV → faster-whisper (modelo tiny/base, CPU) transcreve em português → análise por listas de termos (ansiedade, hesitação, desconforto) → relatório com transcrição e indicadores.

## Configuração

Edite `config.py` para alterar:

- `YOLO_POSE_MODEL` — `yolov8n-pose.pt` (nano) ou `yolov8s-pose.pt` (small).
- `VIDEO_SAMPLE_EVERY_N_FRAMES` — Processar 1 a cada N frames (padrão 3; menor = mais padrões detectados, mais uso de CPU).
- `WHISPER_MODEL_SIZE` — `tiny` ou `base` para CPU.

## Estrutura do projeto

- `main.py` — Orquestrador (CLI).
- `config.py` — Configurações (modelos, pastas).
- `video/` — Pipeline de vídeo (YOLOv8-pose, heurísticas, relatório, vídeo anotado com Pillow e MoviePy).
- `audio/` — Pipeline de áudio (MoviePy, faster-whisper, análise de texto, relatório).
- `data/` — Pasta para vídeo(s) de demonstração.
- `output/` — Relatórios gerados (criada automaticamente).

## Padrão de código

O projeto usa **black**, **isort**, **flake8** e **mypy** (ver `pyproject.toml` e `.pre-commit-config.yaml`). Não altere essas configurações.

## Entregáveis (Tech Challenge Fase 4)

- Análise de vídeo com YOLOv8 (pose) e relatório com indicadores visuais de desconforto e alertas para triagem.
- Processamento de gravação de voz (transcrição + análise do texto) com relatório.
- Execução local em computador pessoal, sem integração com serviços em nuvem.
