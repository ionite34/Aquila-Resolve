import pytest
from Aquila_Resolve.data import remote

# noinspection SpellCheckingInspection
ex_chk = '31544b806735dcdb4bd9aa482339c1ceaf28ca858da12713b5473033c3178899'
# noinspection SpellCheckingInspection
ex_res = '\n'.join(['version',
                    'https://git-lfs.github.com/spec/v1',
                    'oid',
                    'sha256:31544b806735dcdb4bd9aa482339c1ceaf28ca858da12713b5473033c3178899',
                    'size',
                    '111438393'])


def test_check_model(mocker):
    # Mock get_checksum function
    mock_g_ch = mocker.MagicMock(return_value=ex_chk)
    mocker.patch('Aquila_Resolve.data.remote.get_checksum', mock_g_ch)
    assert remote.check_model() is True
    assert mock_g_ch.call_count == 1


def test_check_model_req_fail(mocker):
    # Failure for remote checksum fetching
    mock_get = mocker.MagicMock()
    mock_get.text = mocker.MagicMock(return_value=None)
    mocker.patch('Aquila_Resolve.data.remote.requests.get', mock_get)
    with pytest.warns(UserWarning):
        assert remote.check_model() is False
        assert mock_get.call_count == 1


def test_check_model_fail(mocker):
    # Failure for local checksum mismatch
    mock_g_ch = mocker.MagicMock(return_value='wrong_checksum')
    mocker.patch('Aquila_Resolve.data.remote.get_checksum', mock_g_ch)
    assert check_model() is False
    assert mock_g_ch.call_count == 1


def test_download_existing(mocker):
    # Case for already valid downloaded file
    mock_get = mocker.MagicMock(return_value=None)
    mock_model = mocker.MagicMock()
    mock_model.exists = mocker.MagicMock(return_value=True)
    mocker.patch('Aquila_Resolve.data.remote._model', mock_model)
    mocker.patch('Aquila_Resolve.data.remote.check_model', return_value=True)
    mocker.patch('Aquila_Resolve.data.remote.requests.get', mock_get)
    assert remote.download() is True
    assert mock_get.call_count == 0


def test_ensure_download(mocker):
    mocker.patch('Aquila_Resolve.data.remote.download', return_value=True)
    remote.ensure_download()


def test_ensure_download_fail(mocker):
    mocker.patch('Aquila_Resolve.data.remote.download', return_value=False)
    with pytest.raises(RuntimeError):
        remote.ensure_download()
