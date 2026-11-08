"""Non-dd commands that name a raw block device as their target need approval."""

from __future__ import annotations

import pytest

from tools.approval_detection import detect_dangerous_command


@pytest.mark.parametrize(
    "command",
    (
        "ssdt run --environment real --target /dev/nvme0n1 --yes",
        "ssdt run --target=/dev/sda --environment real --yes",
        "ssdt run --target '/dev/nvme0n1' --yes",
        "ssh host 'sudo ssdt --target /dev/nvme0n1 --yes'",
        "ssdt run --target /dev/disk/by-id/nvme-SSD_1 --yes",
    ),
)
def test_raw_block_device_target_requires_approval(command: str) -> None:
    dangerous, _, description = detect_dangerous_command(command)

    assert dangerous is True, command
    assert "block device" in description


@pytest.mark.parametrize(
    "command",
    (
        "ssdt run --environment real --target results/day1 --yes",
        "ssdt run --target=./report.json --yes",
        "ls -l /dev/nvme0n1",
    ),
)
def test_ordinary_targets_and_block_device_reads_stay_unprompted(
    command: str,
) -> None:
    assert detect_dangerous_command(command) == (False, None, None)
