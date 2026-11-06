"""Tests for the /version slash command."""

from unittest.mock import patch

from cli import HermesCLI






def test_process_command_version_prints_version_info():
    cli_obj = HermesCLI.__new__(HermesCLI)

    with patch("hermes_cli.main._print_version_info") as mock_print:
        assert cli_obj.process_command("/version") is True

    mock_print.assert_called_once_with(check_updates=True)


def test_cli_version_does_not_pay_for_an_implicit_update_check():
    """``hermes --version`` reports local identity; ``--check-updates`` opts into Git/network work."""
    from hermes_cli import main

    args = main.argparse.Namespace(version=True, check_updates=False)

    with patch("hermes_cli.main._print_version_info") as mock_print:
        main.cmd_version(args)

    mock_print.assert_called_once_with(check_updates=False)
