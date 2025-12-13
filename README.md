# Physics-188-Sleep-Studies

Using the DREAMT: Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology, we interpret the data to find possible underlying causes of sleep apnea and possibly more.

We use 9 features: 4 EEG, 1 ECG, 4 Respiratory

Physics features:
input: ['ECG', 'Central_Apnea', 'BVP', 'PTAF', 'THORAX', 'Sleep_Stage']
----> outputs: normalized data frame of features: test, train, val 

features:
#hrv
'RMSSD'
'SDNN'
'mean_HR'

#ptt
'PTT_Mean'
'PTT_STD'

#morphology
'Notch_Complexity'
'Mayer_Power'

#current sleep apnea label
#note: using central apnea
'label'

#resp feats
'Breath_Rate'
'rrSTD'
'Flow_Effort_Corr'
'Flow_Effort_Ratio'
'Flow_Dropout_Fraction'