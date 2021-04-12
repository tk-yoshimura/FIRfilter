import numpy as np
import scipy.signal as signal

class LowPass():
    def __init__(self, sample_hz : float, cutoff_hz : float, numtaps = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, cutoff_hz / nyquist_hz)
        
        self.__sample_hz = sample_hz
        self.__cutoff_hz = cutoff_hz

    def __call__(self, x : np.ndarray, axis = None) -> np.ndarray:
        y = signal.lfilter(self.__firwin, 1, x, -1 if axis is None else axis)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def cutoff_hz(self) -> float:
        return self.__cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin

class HighPass():
    def __init__(self, sample_hz : float, cutoff_hz : float, numtaps = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, cutoff_hz / nyquist_hz, pass_zero=False)
        
        self.__sample_hz = sample_hz
        self.__cutoff_hz = cutoff_hz

    def __call__(self, x : np.ndarray, axis = None) -> np.ndarray:
        y = signal.lfilter(self.__firwin, 1, x, -1 if axis is None else axis)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def cutoff_hz(self) -> float:
        return self.__cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin

class BandPass():
    def __init__(self, sample_hz : float, lower_cutoff_hz : float, higher_cutoff_hz : float, numtaps = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, [lower_cutoff_hz / nyquist_hz, higher_cutoff_hz / nyquist_hz], pass_zero=False)
        
        self.__sample_hz = sample_hz
        self.__lower_cutoff_hz = lower_cutoff_hz
        self.__higher_cutoff_hz = higher_cutoff_hz

    def __call__(self, x : np.ndarray, axis = None) -> np.ndarray:
        y = signal.lfilter(self.__firwin, 1, x, -1 if axis is None else axis)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def lower_cutoff_hz(self) -> float:
        return self.__lower_cutoff_hz

    @property
    def higher_cutoff_hz(self) -> float:
        return self.__higher_cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin

class BandCut():
    def __init__(self, sample_hz : float, lower_cutoff_hz : float, higher_cutoff_hz : float, numtaps = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, [lower_cutoff_hz / nyquist_hz, higher_cutoff_hz / nyquist_hz])
        
        self.__sample_hz = sample_hz
        self.__lower_cutoff_hz = lower_cutoff_hz
        self.__higher_cutoff_hz = higher_cutoff_hz

    def __call__(self, x : np.ndarray, axis = None) -> np.ndarray:
        y = signal.lfilter(self.__firwin, 1, x, -1 if axis is None else axis)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def lower_cutoff_hz(self) -> float:
        return self.__lower_cutoff_hz

    @property
    def higher_cutoff_hz(self) -> float:
        return self.__higher_cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin