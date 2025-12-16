import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert

'''
takes in PTAF/FLOW and THORAX
'''
def analyze_respiratory_mechanics(flow_signal, thorax_signal, sampleRate=100):
    
    # bandpass filter with typical width 0.1-0.5 Hz
    nyquist = 0.5 * sampleRate
    b, a = butter(2, [0.1 / nyquist, 0.5 / nyquist], btype='band')
    
    filtFlow = filtfilt(b, a, flow_signal)
    filtThorax = filtfilt(b, a, thorax_signal)
    
    #RRV
    peaks, _ = find_peaks(filtThorax, distance=sampleRate * 2)
    
    if len(peaks) > 1:
        breath_intervals = np.diff(peaks) / sampleRate 
        
        #std of RRV
        rrv_sd = np.std(breath_intervals)
        
        #avg BPM
        avg_breath_rate = (len(peaks) / (len(flow_signal)/sampleRate)) * 60
    else:
        rrv_sd = 0.0
        avg_breath_rate = 0.0

    # effort flow coupling
    coupling_corr = np.corrcoef(filtFlow, filtThorax)[0, 1]
    
    # low ratio implies obstructive apnea
    flow_energy = np.std(filtFlow)
    effort_energy = np.std(filtThorax)
    coupling_ratio = flow_energy / effort_energy if effort_energy > 0 else 0.0


    # ahi surrogate
    # Physics: Apnea is >90% drop in flow for >10 seconds.
    analytic_signal = hilbert(filtFlow)
    amplitude_envelope = np.abs(analytic_signal)
    baseline = np.median(amplitude_envelope)
    dropout_threshold = baseline * 0.10
    
    # samples where flow is gone
    dropout_samples = amplitude_envelope < dropout_threshold
    
    # percentage of dropouts
    # should strongly correlate with our central apnea
    dropout_fraction = np.sum(dropout_samples) / len(dropout_samples)
    
    return {
        'RRV_SD': rrv_sd,
        'Breath_Rate': avg_breath_rate,
        'Flow_Effort_Corr': coupling_corr,
        'Flow_Effort_Ratio': coupling_ratio,
        'Flow_Dropout_Fraction': dropout_fraction
    }