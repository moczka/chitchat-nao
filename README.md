# Chitchat NAO

Chitchat NAO is a laptop-hosted assistant for a NAO6 robot. It uses [SmolLM2-1.7B](https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/blob/main/SmolLM2-1.7B-Instruct-Q6_K.gguf) as its small language model.

The system design is documented in [docs/architecture.md](docs/architecture.md).

## Development setup

The project targets Python 3.13 or newer, uses [uv](https://docs.astral.sh/uv/) for package management, and uses Ruff for linting.

```sh
uv sync
source .venv/bin/activate
```

## Text-to-speech bridge

[`say_nao.py`](say_nao.py) is a Python 3 standard-library launcher. It finds the bundled `qicli` executable and prepares a call to `ALTextToSpeech.say`. Its defaults are `ROBOT_IP=192.168.50.33` and `ROBOT_PORT=9561`; these can also be set through the environment.

For offline work, use dry-run mode. It prints the resolved command without contacting a robot:

```sh
uv run python say_nao.py --dry-run 'Hello from the laptop'
```

The bridge also accepts `--robot-ip`, `--port`, and `--qicli` to override the defaults or qicli path. Do not use the non-dry-run form unless a robot is configured and available.

## Offline verification

Run the lint and test checks locally with:

```sh
uv run ruff check say_nao.py tests
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python say_nao.py --dry-run 'Hello from the laptop'
```

## Project layout

- `say_nao.py`: Python 3 qicli text-to-speech bridge.
- `tests/`: offline tests for the bridge.
- `docs/architecture.md`: system architecture and design principles.

`say_nao_py2.py` and `Dockerfile.naoqi-py2` are legacy Python 2/direct-NAOqi experiments, not the normal development workflow.
