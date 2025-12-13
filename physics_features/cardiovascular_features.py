import numpy as np
from scipy.signal import find_peaks
from scipy.signal import welch

'''
Uses ECG and BVP
'''
def hrVariability(ecgData, sampleRate=100, peakDistance=0.5):

    #R peaks
    #change height? avg is good enough
    peaks, _ = find_peaks(ecgData, distance=int(sampleRate * peakDistance), height=np.mean(ecgData))
    
    rrIntervals = np.diff(peaks) / sampleRate * 1000 
    
    # SDNN (Standard Deviation of NN intervals)
    # overall variability of heart rate
    # significantly lower SDNN often indicates that the heart's natural variability is being suppressed
    sdnn = np.std(rrIntervals, ddof=1)
    
    # RMSSD: (Root Mean Square of Successive Differences) 
    # short term variability --> looks at adjacent beats
    # RMMSD = \sqrt{  1/(N-1) \sum{ RR_i+1 -RR_i}^2}
    rrDiff = np.diff(rrIntervals)
    rmssd = np.sqrt(np.mean(rrDiff ** 2))
    
    return sdnn, rmssd, peaks

# note the following functions use blood volume pulse to find transit times
# returns mean & std of pulsed transit time
def PTT(ecgPeaks, bvp, sampleRate=100):

    transitTimes = []
    
    # range of ptt is around 200-400 ms
    lower = int(0.200 * sampleRate)
    upper = int(0.400 * sampleRate)
    
    for rPeak in ecgPeaks:
        
        #search BVP signal at peaks 
        start_search = rPeak + lower
        end_search = rPeak + upper
        
        if end_search < len(bvp):
            window = bvp[start_search : end_search]
            
            # Find the max peak in this window (systolic peak)
            if len(window) > 0:
                local_peak_idx = np.argmax(window)
                bvp_peak_idx = start_search + local_peak_idx
                
                # PTT = Time of BVP Peak - Time of R Peak
                ptt_samples = bvp_peak_idx - rPeak
                ptt_ms = (ptt_samples / sampleRate) * 1000
                transitTimes.append(ptt_ms)
                
    return np.mean(transitTimes), np.std(transitTimes)

# dicrotic notch caused by the closing of the aortic valve
# the notch refers to the brief backward flow in blood 
# asseses vascular stiffness 
def dnAnalysis(bvpData):
    #dicroticNotches = []

    # find 2nd derivative to find slope/notch
    deriv1 = np.gradient(bvpData)
    deriv2 = np.gradient(deriv1)

    morphology = np.var(deriv2)
    return morphology

# mayer wave detection
def mayerWave(bvpData, sampleRate=100):

    # Power Spectral Density
    cleanBvp = bvpData.to_numpy().flatten()
    f, Pxx = welch(cleanBvp, fs=sampleRate, nperseg=sampleRate*90) 
    
    # envelope ~0.1Hz
    mayer_band = (f >= 0.05) & (f <= 0.15)
    
    mayer_power = np.trapz(Pxx[mayer_band], f[mayer_band])
    
    if mayer_power < 1e-9:
        return 0.0
    # normalize
    total_power = np.trapz(Pxx, f)
    return mayer_power / total_power