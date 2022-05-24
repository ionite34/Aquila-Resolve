# CLI Entry Point
import Aquila_Resolve
from InquirerPy import inquirer
from InquirerPy.utils import color_print as cp
from yaspin import yaspin


def main_menu():
    """
    Aquila Resolve Entry Point
    """
    g2p_convert()
    exit(0)


def g2p_convert():  # pragma: no cover
    """
    G2P Conversion
    """
    with yaspin('Initializing Aquila Resolve Backend...', color='yellow') as sp:
        g2p = Aquila_Resolve.G2p()
        sp.ok(f'✔ Aquila Resolve v{Aquila_Resolve.__version__}')

    while True:
        text = inquirer.text("Text to convert:").execute()
        if not text:
            return
        result = g2p.convert(text)
        cp([('yellow', f'{result}')])
