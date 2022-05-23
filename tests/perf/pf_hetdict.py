# Het dict performance testing

from utils import catch_time
from Aquila_Resolve.dictionary import Dictionary
from Aquila_Resolve.het_dict import HetDict
from Aquila_Resolve import G2p


def run():
    with catch_time('Dictionary Init'):
        d = Dictionary()
    with catch_time('HetDict Init'):
        hd = HetDict()
    with catch_time('G2p Init'):
        g2p = G2p()


if __name__ == '__main__':
    run()
