import argparse
import tempfile
import unittest
from pathlib import Path

from say_nao import (
    build_command,
    normalize_text,
    parse_port,
    resolve_qicli,
)


class SayNaoTests(unittest.TestCase):
    def test_parse_port_accepts_valid_port(self):
        self.assertEqual(parse_port("9561"), 9561)

    def test_parse_port_rejects_non_integer(self):
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "Expected an integer"
        ):
            parse_port("not-a-port")

    def test_parse_port_rejects_values_outside_range(self):
        for value in ("0", "65536"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError, "1 to 65535"
                ):
                    parse_port(value)

    def test_normalize_text_collapses_whitespace(self):
        self.assertEqual(normalize_text("  Hello\tNAO\n  "), "Hello NAO")

    def test_normalize_text_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "empty speech text"):
            normalize_text(" \t\n ")

    def test_build_command_returns_qicli_arguments(self):
        command = build_command(
            Path("/tmp/qicli"), "tcp://192.168.50.33:9561", "Hello NAO"
        )
        self.assertEqual(
            command,
            [
                "/tmp/qicli",
                "call",
                "ALTextToSpeech.say",
                "Hello NAO",
                "--qi-url",
                "tcp://192.168.50.33:9561",
            ],
        )

    def test_resolve_qicli_finds_temporary_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "qicli"
            executable.touch()
            executable.chmod(executable.stat().st_mode | 0o111)

            self.assertEqual(resolve_qicli(executable), executable)

    def test_resolve_qicli_rejects_missing_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-qicli"
            with self.assertRaisesRegex(FileNotFoundError, "--qicli"):
                resolve_qicli(missing)


if __name__ == "__main__":
    unittest.main()
