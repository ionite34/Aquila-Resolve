# Performance Test Utilities
import time


class catch_time:
    def __enter__(self):
        self.time = time.perf_counter_ns()
        return self

    def __exit__(self, ex_type, ex_value, traceback):
        self.time = time.perf_counter_ns() - self.time
        self.readout = f"Time: {(self.time / 1e6):.4f} ms"
        print(self.readout)
