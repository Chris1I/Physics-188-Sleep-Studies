import cardiovascular_features as cvf
import respiratory_features as rpf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob

def featureCreate(fileName, inputColumns=['ECG', 'Central_Apnea', 'BVP', 'PTAF', 'THORAX', 'Sleep_Stage']):

    #data = pd.read_csv(r'dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0/data_100Hz/S007_PSG_df_updated.csv'
    #                , usecols=['ECG', 'Central_Apnea', 'BVP', 'PTAF', 'THORAX'])
    data = pd.read_csv(fileName, usecols=inputColumns)

    sleepStageData = data['Sleep_Stage']

    ecgData = data['ECG']
    bvpData = data['BVP']

    ptaData = data['PTAF']
    thoData = data['THORAX']

    #clean binary
    data['Central_Apnea'] = data['Central_Apnea'].fillna(0).astype(int)
    apneaLabels = data['Central_Apnea']
    central_apnea_df = data[data['Central_Apnea'] == 1]
    #print(f"Found {len(central_apnea_df)} Central Apnea events. out of {len(apneaLabels)} {len(central_apnea_df) / len(apneaLabels) * 100}%")
    #print(data.head())

    seconds = 90 
    stepSeconds = 30
    sampleRate = 100 # assuming 100Hz is used

    #x sec * (100 samples / 1 sec)
    windowSample = seconds * sampleRate 
    stepSample = stepSeconds * sampleRate #3000

    #epochs = len(ecgData) // windowSample 
    epochs = (len(ecgData) - windowSample) // stepSample 
    print("epochs: ", epochs)
    featuresList = []

    for i in range(epochs):

        start = i * stepSample
        end = start + windowSample

        #slicing
        sleepStageSignal = sleepStageData[start:end]
        ecgSignal = ecgData[start:end]
        bvpSignal = bvpData[start:end]

        ptaSignal = ptaData[start:end]
        thoSignal = thoData[start:end]
        labelWindow = apneaLabels[start:end]

        #check for dead signals **not necessary
        #if np.std(bvpSignal) == 0 or \
        #   np.isnan(bvpSignal).any():
        #    print("in")
        #    continue

        #filter out sleep stage preparation, missing, wake
        if np.any(sleepStageSignal == 'P') \
           or np.any(sleepStageSignal == 'Missing'):
            #print("in")
            continue

        #concatenate to 350 --> 350 * 7 = 2450
        if len(featuresList) >= 350:
            break

        #cardiovascular feature functions
        sdnn, rmssd, peaks = cvf.hrVariability(ecgSignal, sampleRate=sampleRate)
        bpm = (len(peaks) / seconds) * 60
        ttMean, ttStd = cvf.PTT(peaks, bvpSignal, sampleRate=sampleRate)
        morphology = cvf.dnAnalysis(bvpSignal)
        normMayerWavePower = cvf.mayerWave(bvpSignal, sampleRate=sampleRate)

        #respiratory feature function
        #using PTAF instead of FLOW
        respFeat = rpf.analyze_respiratory_mechanics(ptaSignal, thoSignal, sampleRate=sampleRate)

        featuresList.append({
                    #hrv
                    'RMSSD': rmssd,
                    'SDNN': sdnn,
                    'mean_HR': bpm,

                    #ptt
                    'PTT_Mean': ttMean,
                    'PTT_STD': ttStd,

                    #morphology
                    'Notch_Complexity': morphology,
                    'Mayer_Power': normMayerWavePower,

                    #current sleep apnea label
                    #note: using central apnea
                    'label': np.max(labelWindow),

                    #resp feats
                    'Breath_Rate': respFeat['Breath_Rate'],
                    'rrSTD': respFeat['RRV_SD'],
                    'Flow_Effort_Corr': respFeat['Flow_Effort_Corr'],
                    'Flow_Effort_Ratio': respFeat['Flow_Effort_Ratio'],
                    'Flow_Dropout_Fraction': respFeat['Flow_Dropout_Fraction'],
                    })
        

    #--------
    dfFeatures = pd.DataFrame(featuresList)
    ##print(dfFeatures.head())
    #print("features")
    #print(dfFeatures.sample(n=10))
    #print(dfFeatures.shape)
    #
    ##print(dfFeatures[dfFeatures['Mayer_Power'].notna()].head())
    #apnea_events = dfFeatures[dfFeatures['label'] == 1]
    #
    #print(f"Total Apnea Events found: {len(apnea_events)}")
    #print(apnea_events.head(10))
    #
    #print("--- Apnea Event Statistics ---")
    #print(apnea_events.describe())


    #convert dfFeatures to tensor for dense layer
    x = dfFeatures.drop(columns=['label'])
    y = dfFeatures['label'].values

    #fill in missing values with mean or zero
    x = x.fillna(x.mean())
    #x = x.fillna(0)

    #inf values if any
    x = x.replace([np.inf, -np.inf], 0)

    

    return x, y

#make sure to edit this to point to where 100Hz data is if running on own 
file_list = sorted(glob.glob('dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0/data_100Hz/*_PSG_df_updated.csv'))
target_files = file_list[:11]
#target_files = file_list[:34]
print(f"# of people: {len(file_list)}")

X_train_list, Y_train_list = [], []
X_test_list, Y_test_list = [], []
X_val_list, Y_val_list = [], []

#trainers = [2,3,5,6,7,17,21]
#validators = [22,23]
#testers = [24,28]

#targetPeople = []
#targetPeople.extend(trainers)
#targetPeople.extend(validators)
#targetPeople.extend(testers)

#using first 11 people 
for i, filePath in enumerate(target_files):
#for i, filePath in enumerate(target_files):
    person = i + 1
    #SID = i + 2
    #if SID not in targetPeople:
    #    continue 
    x, y = featureCreate(filePath)
    #print('finished SID #', SID)

    #if SID in testers:
    #    X_test_list.append(x)
    #    Y_test_list.append(y)
    #elif SID in validators: 
    #    X_val_list.append(x)
    #    Y_val_list.append(y)
    #elif SID in trainers: 
    #    X_train_list.append(x)
    #    Y_train_list.append(y)


    if person in [8, 9]:
        X_test_list.append(x)
        Y_test_list.append(y)
    elif person in [10, 11]: 
        X_val_list.append(x)
        Y_val_list.append(y)
    else:
        X_train_list.append(x)
        Y_train_list.append(y)

#concatenate
X_train_df = pd.concat(X_train_list, ignore_index=True)
Y_train = np.concatenate(Y_train_list, axis=0)

X_test_df = pd.concat(X_test_list, ignore_index=True)
Y_test = np.concatenate(Y_test_list, axis=0)

X_val_df = pd.concat(X_val_list, ignore_index=True)
Y_val = np.concatenate(Y_val_list, axis=0)

#normalize for gradient descent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)
X_val_scaled = scaler.transform(X_val_df)

print("\n--- Final Shapes ---")
print(f"Train Matrix: {X_train_scaled.shape}")
print(f"Test Matrix:  {X_test_scaled.shape}")
print(f"Validation Matrix:  {X_val_scaled.shape}")

# Save
np.save('X_train_features.npy', X_train_scaled)
np.save('Y_train_labels.npy', Y_train)
np.save('X_test_features.npy', X_test_scaled)
np.save('Y_test_labels.npy', Y_test)
np.save('X_val_features.npy', X_val_scaled)
np.save('Y_val_labels.npy', Y_val)

