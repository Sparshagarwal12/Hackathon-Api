from fastapi import FastAPI
import numpy as np
from scipy import fftpack

app = FastAPI()

def sample_signal(t):
    s1 = 2 * np.sin(2 * 2 *np.pi * t)
    s2 = 3 * np.sin(22 * 2 * np.pi * t)
    s3 = 2*np.random.randn(*np.shape(t))
    return s1 + s2 + s3

def get_signal(n_s=2048, s_f=2000):
    """n_s : no of sample
       s_f : sampling frequency
    """
    s_p = n_s / s_f  # sampling period
    t = np.linspace(0, s_p, n_s)
    signal = sample_signal(t)
    return {"time": t.tolist(), "signal": signal.tolist()}

def get_fft(n_s=2048, s_f=2000, max_freq=2000, log=False):
    """Return the fft """
    s_p = n_s / s_f  # sampling period
    t = np.linspace(0, s_p, n_s)
    signal = sample_signal(t)
    F = fftpack.fft(signal)
    f = fftpack.fftfreq(n_s, 1.0/s_f)
    f1 = np.nonzero(f>0)
    f1 = np.nonzero(f[f1] <= max_freq)
    freq_ax = f[f1]
    fft_ax = abs(F[f1]/n_s)
    if log==True:
        return {"freq": freq_ax.tolist(), "fft": np.log(fft_ax).tolist()}
    else:
        return {"freq": freq_ax.tolist(), "fft": fft_ax.tolist()}


def get_dominant_freq(n_s=6000, s_f=2000, max_freq=2000):
    F = get_fft(n_s=n_s, s_f=s_f, max_freq=max_freq)
    fft = np.array(F["fft"])
    freq = np.array(F["freq"])
    return freq[np.argmax(fft)]

@app.get("/")
def root():
    return {"message":"Welcome to laser vibrometer interface" }


@app.get("/signal/")
def signal(sampfreq: int=2048, sampnum: int=2000):
    a = get_signal(n_s=sampnum, s_f=sampfreq)
    return a

@app.get("/fft/")
def fft(sampfreq: int=2048, sampnum: int=2000, maxfreq: int=2000):
    f = get_fft(n_s=sampnum, s_f=sampfreq, max_freq=maxfreq)
    return f

@app.get("/dom/")
def dom_freq(sampfreq: int=2048, sampnum: int=6000, maxfreq: int=2000):
    f = get_dominant_freq(n_s=sampnum, s_f=sampfreq, max_freq=maxfreq)
    return f


@app.get("/fft/log")
def log_fft():
    f = get_fft(log=True)
    return f


@app.get("/fft/{max_freq}")
def fft_max(max_freq):
    f = get_fft(max_freq=int(max_freq))
    return f
