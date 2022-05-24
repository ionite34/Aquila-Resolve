# Access and checks for remote data
import requests
import shutil
import nltk
from warnings import warn
from tqdm.auto import tqdm
from . import DATA_PATH

_model = DATA_PATH.joinpath('model.pt')
_model_url = "https://huggingface.co/ionite/Aquila-Resolve/resolve/main/model.pt"  # Download URL
_model_ptr = "https://huggingface.co/ionite/Aquila-Resolve/raw/main/model.pt"  # Git LFS Pointer URL


def check_model() -> bool:
    """Checks if the model matches checksums"""
    result = requests.get(_model_ptr).text.split()
    if result is None or len(result) < 6 or not result[3].startswith('sha256:'):
        warn("Could not retrieve remote model checksum")
        return False
    remote_sha256 = result[3][7:]
    actual_sha256 = get_checksum(_model)
    return remote_sha256 == actual_sha256


def download(update: bool = True) -> bool:
    """
    Downloads the model

    :param update: True for download if checksum does not match, False for download if no file exists
    :return: Existence or checksum match of model file
    """
    # Check if the model is already downloaded
    if not update and _model.exists():
        return True
    if update and _model.exists() and check_model():
        return True
    # Download the model
    with requests.get(_model_url, stream=True) as r:
        r.raise_for_status()  # Raise error for download failure
        total_size = int(r.headers.get('content-length', 0))
        with tqdm.wrapattr(r.raw, 'read', total=total_size, desc='Downloading model checkpoint') as raw:
            with _model.open('wb') as f:
                shutil.copyfileobj(raw, f)
    if update:
        return _model.exists() and check_model()  # Update flag, verify checksum also
    return _model.exists()  # For no update flag, just check existence


def ensure_download() -> None:
    """Ensures the model is downloaded"""
    if not download(update=False):
        raise RuntimeError("Model could not be downloaded. Visit "
                           "https://huggingface.co/ionite/Aquila-Resolve/blob/main/model.pt "
                           "to download the model checkpoint manually and place it within the "
                           "Aquila_Resolve/data/ folder.")


def ensure_nltk() -> None:  # pragma: no cover
    """Ensures all required NLTK Data is installed"""
    required = {
        'wordnet': 'corpora/wordnet.zip',
        'omw-1.4': 'corpora/omw-1.4.zip',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger.zip',
    }
    for name, url in required.items():
        try:
            nltk.data.find(url)
        except LookupError:
            nltk.download(name, raise_on_error=True)


def check_updates() -> None:
    """Checks if the model matches the latest checksum"""
    if not check_model():
        warn("Local model checkpoint did not match latest remote checksum. "
             "You can use Aquila_Resolve.download() to download the latest model.")


def get_checksum(file: str, block_size: int = 65536) -> str:
    """
    Calculates the Sha256 checksum of a file

    :param file: Path to file
    :param block_size: Block size for reading
    :return: Checksum of file
    """
    import hashlib
    s = hashlib.sha256()
    with open(file, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            s.update(block)
    return s.hexdigest()
