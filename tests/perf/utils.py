# Performance Test Utilities
import time
from yaspin import yaspin


class catch_time:
    def __init__(self, desc: str):
        self.time = None
        self.readout = None
        self.desc = desc
        self.sp = None

    def __enter__(self):
        self.sp = yaspin(text=self.desc, color='yellow')
        self.sp.start()
        self.time = time.perf_counter_ns()
        return self

    def __exit__(self, ex_type, ex_value, traceback):
        self.time = time.perf_counter_ns() - self.time
        self.readout = self.printout()
        # print(self.readout)
        self.sp.ok(f'✔ {self.readout}')

    def printout(self) -> str:
        # Check if ns is over 1 s
        if self.time >= 1e9:
            return f'{(self.time / 1e9):.2f} s'
        # Check if ns is over 1 ms
        elif self.time >= 1e6:
            return f'{(self.time / 1e6):.2f} ms'
        # Check if ns is over 1 μs
        elif self.time >= 1000:
            return f'{(self.time / 1000):.2f} μs'
        # Check if ns is over 1 ns
        elif self.time >= 1:
            return f'{self.time:.2f} ns'
        # Else
        else:
            return f'{self.time} ns'
