import numpy as np
import scipy.signal as signal

class FIRFilter():
    @staticmethod
    def filtering(x, firwin, axis):
        axis = x.ndim - 1 if axis is None else axis

        pn = len(firwin)
        dn = x.shape[axis]

        if axis == 0:
            s = [slice(1, pn + 1)] + [slice(None)] * (x.ndim - 1)
        elif axis == x.ndim - 1:
            s = [slice(None)] * (x.ndim - 1) + [slice(1, pn + 1)]
        else:
            s = [slice(None)] * (axis) + [slice(1, pn + 1)] + [slice(None)] * (x.ndim - axis - 1)
        
        s = tuple(s)

        x_edge = x[s]

        if axis == 0:
            s = [slice(0, 1)] + [slice(None)] * (x.ndim - 1)
        elif axis == x.ndim - 1:
            s = [slice(None)] * (x.ndim - 1) + [slice(0, 1)]
        else:
            s = [slice(None)] * (axis) + [slice(0, 1)] + [slice(None)] * (x.ndim - axis - 1)

        s = tuple(s)

        x0 = x[s]

        x_reflact = 2 * x0 - np.flip(x_edge, axis)

        x_concat = np.concatenate([x_reflact, x], axis)

        y = signal.lfilter(firwin, 1, x_concat, axis)

        if axis == 0:
            s = [slice(pn, None)] + [slice(None)] * (x.ndim - 1)
        elif axis == x.ndim - 1:
            s = [slice(None)] * (x.ndim - 1) + [slice(pn, None)]
        else:
            s = [slice(None)] * (axis) + [slice(pn, None)] + [slice(None)] * (x.ndim - axis - 1)

        s = tuple(s)

        y_slice = y[s]

        return y_slice

class LowPass():
    def __init__(self, sample_hz : float, cutoff_hz : float, numtaps = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, cutoff_hz / nyquist_hz)
        
        self.__sample_hz = sample_hz
        self.__cutoff_hz = cutoff_hz

    def __call__(self, x : np.ndarray, axis = None) -> np.ndarray:
        y = FIRFilter.filtering(x, self.__firwin, axis)

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
        y = FIRFilter.filtering(x, self.__firwin, axis)

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
        y = FIRFilter.filtering(x, self.__firwin, axis)

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
        y = FIRFilter.filtering(x, self.__firwin, axis)

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