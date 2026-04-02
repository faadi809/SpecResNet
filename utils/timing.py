import time
import torch

def measure_time(fn, runs=25):
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            fn()
            torch.cuda.synchronize()
            times.append(time.time() - start)
    return sum(times) / len(times)