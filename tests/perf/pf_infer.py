# Infer performance testing
import time

from yaspin import yaspin

from utils import catch_time
from Aquila_Resolve import G2p
from Aquila_Resolve.infer import Infer
from Aquila_Resolve.resolve import Resolve

_TARGET = "F:/Repos/Python/test-data/nora.csv"


def run():
    # prepare the test data
    t_lines = []
    with catch_time('Load Test Data'):
        with open(_TARGET, 'r') as f:
            for line in f:
                t_lines.append(line.split('|')[1])

    # Randomly select 100 lines from the test data
    t_lines = t_lines[:1000]

    with catch_time('G2p Init'):
        g2p = G2p(device='cuda')
        g2p.convert('Bravalagos')

    with catch_time('Resolve Init'):
        resolve = Resolve(device='cuda')
        resolve(['Bravalagos in Etephyrus.'])

    # Wait 5 seconds for warmup
    with yaspin('Waiting 5 seconds...', color='yellow', timer=True) as sp:
        time.sleep(5)
        sp.ok('✔ Warmup Done')

    with catch_time('G2p'):
        for line in t_lines:
            g2p.convert(line.lower())

    with catch_time('Resolve'):
        resolve(t_lines)

    with catch_time('Phonemise_list Single'):
        pred = g2p.infer.phonemizer.phonemise_list(t_lines, lang='en_us')

    with catch_time('Phonemise_list Loop'):
        for line in t_lines:
            g2p.infer.phonemizer.phonemise_list([line], lang='en_us')

    # with catch_time('Infer Method'):
    # for line in t_lines:
    # g2p.infer([line])


def infer_runs():
    from nltk import TweetTokenizer
    tkn = TweetTokenizer()
    # prepare the test data
    t_lines = []
    with catch_time('Load Test Data'):
        with open(_TARGET, 'r') as f:
            for line in f:
                t_lines.append(line.split('|')[1])

    # Randomly select 200 lines from the test data
    t_lines = t_lines[:200]

    with catch_time('G2p Init'):
        g2p = G2p(device='cuda')

    with catch_time('Phonemise_list Single'):
        g2p.infer.phonemizer.phonemise_list(t_lines, lang='en_us')

    with catch_time('Phonemise_list Loop'):
        for line in t_lines:
            g2p.infer.phonemizer.phonemise_list([line], lang='en_us')


if __name__ == '__main__':
    run()
