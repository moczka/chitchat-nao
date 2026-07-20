from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.50.33")
DEFAULT_PORT = os.environ.get("ROBOT_PORT", "9561")
COMMAND_TIMEOUT_SECONDS = 15
PROJECT_ROOT = Path(__file__).resolve().parent
QICLI_CANDIDATES = (
    PROJECT_ROOT / ".local" / "naoqi-sdk" / "bin" / "qicli",
    PROJECT_ROOT / ".local" / "pynaoqi" / "bin" / "qicli",
    PROJECT_ROOT
    / ".local"
    / "pynaoqi-extracted"
    / "pynaoqi-python2.7-2.8.7.4-linux64-20210819_141148"
    / "bin"
    / "qicli",
)


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid port '{value}'. Expected an integer."
        ) from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"Invalid port '{value}'. Expected a value from 1 to 65535."
        )
    return port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send text-to-speech to a NAO robot via the bundled qicli runtime."
        )
    )
    try:
        default_port = parse_port(DEFAULT_PORT)
    except argparse.ArgumentTypeError as error:
        parser.error(f"Invalid ROBOT_PORT environment variable: {error}")

    parser.add_argument(
        "text",
        nargs="?",
        default="Hello. This is the laptop controlling NAO.",
        help="Text for NAO to say.",
    )
    parser.add_argument(
        "--robot-ip",
        default=DEFAULT_ROBOT_IP,
        help="NAO robot IP address.",
    )
    parser.add_argument(
        "--port",
        type=parse_port,
        default=default_port,
        help="NAOqi port. Project default is 9561.",
    )
    parser.add_argument(
        "--qicli",
        type=Path,
        help="Explicit path to qicli if auto-detection is unsuitable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the qicli command without executing it.",
    )
    return parser.parse_args()


def resolve_qicli(explicit_path: Path | None) -> Path:
    candidates = (explicit_path,) if explicit_path else QICLI_CANDIDATES
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    message = f"Could not find an executable qicli. Checked:\n  - {searched}\n"
    if explicit_path:
        message += (
            "The path provided via --qicli was not found or is not executable."
        )
    else:
        message += "Pass --qicli to override the path."
    raise FileNotFoundError(message)


def normalize_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("Refusing to send empty speech text.")
    return normalized


def build_command(qicli_path: Path, qi_url: str, text: str) -> list[str]:
    return [
        str(qicli_path),
        "call",
        "ALTextToSpeech.say",
        text,
        "--qi-url",
        qi_url,
    ]


def run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    print("qicli command completed successfully.")


def main() -> int:
    args = parse_args()
    try:
        qicli_path = resolve_qicli(args.qicli)
        text = normalize_text(args.text)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    command = build_command(
        qicli_path=qicli_path,
        qi_url=f"tcp://{args.robot_ip}:{args.port}",
        text=text,
    )
    if args.dry_run:
        print("Resolved qicli command:")
        print(shlex.join(command))
        return 0

    try:
        run_command(command)
    except subprocess.CalledProcessError as error:
        print(
            f"qicli failed with exit code {error.returncode}.", file=sys.stderr
        )
        if error.stdout:
            print(error.stdout.strip(), file=sys.stderr)
        if error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        return error.returncode
    except subprocess.TimeoutExpired:
        print("qicli timed out while contacting the robot.", file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        print("Interrupted before qicli completed.", file=sys.stderr)
        return 130
    except OSError as error:
        print(f"Failed to launch qicli: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
