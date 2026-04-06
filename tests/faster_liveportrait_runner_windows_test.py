from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import faster_liveportrait_runner


def test_process_pid_is_alive_decodes_tasklist_bytes() -> None:
    completed = SimpleNamespace(stdout="python.exe           1234 Consola                    1     10.240 K\n".encode("cp1252"))
    with patch("faster_liveportrait_runner.subprocess.run", return_value=completed):
        assert faster_liveportrait_runner.process_pid_is_alive(1234) is True


def test_process_pid_is_alive_handles_non_utf8_tasklist_output() -> None:
    completed = SimpleNamespace(stdout="Informaci\xf3n del proceso python.exe PID 1234".encode("cp1252"))
    with patch("faster_liveportrait_runner.subprocess.run", return_value=completed):
        assert faster_liveportrait_runner.process_pid_is_alive(1234) is True


def main() -> None:
    test_process_pid_is_alive_decodes_tasklist_bytes()
    test_process_pid_is_alive_handles_non_utf8_tasklist_output()
    print("ok")


if __name__ == "__main__":
    main()
