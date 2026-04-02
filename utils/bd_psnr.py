import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad

def bd_psnr(rate1, psnr1, rate2, psnr2):
    log_r1 = np.log(rate1)
    log_r2 = np.log(rate2)

    f1 = PchipInterpolator(log_r1, psnr1)
    f2 = PchipInterpolator(log_r2, psnr2)

    low = max(min(log_r1), min(log_r2))
    high = min(max(log_r1), max(log_r2))

    integral = quad(lambda x: f2(x) - f1(x), low, high)[0]
    return integral / (high - low)
