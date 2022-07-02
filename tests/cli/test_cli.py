import pytest
from Aquila_Resolve.cli import cli


def test_main_menu(mocker):
    mocker.patch.object(cli, "g2p_convert", return_value=None)
    with pytest.raises(SystemExit):
        cli.main_menu()
