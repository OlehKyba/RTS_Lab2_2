import numpy as np
import matplotlib.pyplot as plt

from random import random
from math import sin, cos, pi

from functools import wraps
from time import time

N = 256 #number of discrete
n = 8 #number of harmonic
w = 2000 #cutoff frequency

x = np.arange(0, N, 1)


def spectrum_wrapper(func):
    @wraps(func)
    def inner_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        real, image = np.array([i.real for i in result]), np.array([i.imag for i in result])
        return real, image
    return inner_wrapper


def time_wrapper(func):
    @wraps(func)
    def inner_wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        time_delta = start - time()
        return result, time_delta
    return inner_wrapper


def generate_random_signals(steps=N, max_a=5):
    def generate_random_signal(t):
        sum_res = 0
        wp = w / n

        for i in range(n):
            a = max_a * random()
            fi = 2 * pi * random()
            sum_res += a * sin(wp * t + fi)
            wp += w / n

        return sum_res

    x_array = np.array([generate_random_signal(i) for i in range(steps)])
    return x_array


@spectrum_wrapper
def dft(signal):
    def factor(pk, n):
        angle = -2 * pi / n * pk
        return complex(cos(angle), sin(angle))

    length = len(signal)
    result = np.zeros(length, dtype=complex)

    for p in range(length):
        for k in range(length):
            result[p] += factor(p * k, length)*signal[k]

    return result


@spectrum_wrapper
def fft(signal):
    def factor(pk, n):
        angle = -2 * pi / n * pk
        return complex(cos(angle), sin(angle))
    
    def inner_fft(signal, p, level_factor):
        n = len(signal)
        next_n = n // 2
        next_p = p % next_n
        if n > 2:
            signal_odd = np.array([signal[i] for i in range(1, n) if i % 2 == 1])
            signal_pair = np.array([signal[i] for i in range(n) if i % 2 == 0])

            next_factor = factor(next_p, next_n)
            f_odd = inner_fft(signal_odd, next_p, next_factor)
            f_pair = inner_fft(signal_pair, next_p, next_factor)
            return f_pair + level_factor * f_odd
        
        w_odd = -1 if p % 2 else 1
        return signal[0] + signal[1] * w_odd
    
    length = len(signal)
    result = np.array([inner_fft(signal, p, factor(p, length)) for p in range(length)])
    return result


def main():
    random_signal = generate_random_signals()

    real_spectrum, image_spectrum = fft(random_signal)
    fig, axes = plt.subplots(3, sharex=True, figsize=(15, 15))
    axes[0].set_title("Signal")
    axes[0].plot(x, random_signal)

    axes[1].set_title("FFT: Real part")
    axes[1].bar(x, real_spectrum)

    axes[2].set_title("FFT: Imaginary part")
    axes[2].bar(x, image_spectrum)

    plt.savefig("lab_2_2.png")
    plt.show()


def extra_task(start=1, finish=11):
    dft_with_time = time_wrapper(dft)
    fft_with_time = time_wrapper(fft)

    def timeshift(n):
        random_signal = generate_random_signals(n)
        _, dft_time = dft_with_time(random_signal)
        _, fft_time = fft_with_time(random_signal)
        return dft_time - fft_time

    m = np.arange(start, finish)
    n = 2**m
    time_deletes = np.array([timeshift(i) for i in n])

    fig, axes = plt.subplots(2, figsize=(15, 15))

    axes[0].plot(m, time_deletes)
    axes[0].set(xlabel='m', ylabel='time (s)', title='Залежність дельти часу від 2**m довжини масиву.')

    axes[1].plot(n, time_deletes)
    axes[1].set(xlabel='n', ylabel='time (s)', title='Залежність дельти часу від n довжини масиву.')

    plt.savefig("lab_2_2_extra.png")
    plt.show()


if __name__ == '__main__':
    main()
    extra_task()
