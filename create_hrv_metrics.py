import pandas as pd
import numpy as np
import wfdb
import ast
import wfdb.processing
import biosppy
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
import pyhrv.time_domain as td
import time
from joblib import Parallel, delayed
try:
    import winsound
    usewin = True
except ImportError:
    usewin = False

nrows=1_000 # number of patients to load, set to None to load all patients
path="./"
sampling_rate=500

# hrv_columns = ['hr_mean','sdnn','rmssd','pnn20','fft_ratio','sd_ratio']
hrv_columns = ['lead1_sdnn', 'lead1_rmssd', 'lead1_fft_ratio', 'lead2_sdnn',
       'lead2_rmssd', 'lead2_fft_ratio', 'lead3_sdnn', 'lead3_rmssd',
       'lead3_fft_ratio', 'lead4_sdnn', 'lead4_rmssd', 'lead4_fft_ratio',
       'lead5_sdnn', 'lead5_rmssd', 'lead5_fft_ratio', 'lead6_sdnn',
       'lead6_rmssd', 'lead6_fft_ratio', 'lead7_sdnn', 'lead7_rmssd',
       'lead7_fft_ratio', 'lead8_sdnn', 'lead8_rmssd', 'lead8_fft_ratio',
       'lead9_sdnn', 'lead9_rmssd', 'lead9_fft_ratio', 'lead10_sdnn',
       'lead10_rmssd', 'lead10_fft_ratio', 'lead11_sdnn', 'lead11_rmssd',
       'lead11_fft_ratio', 'lead12_sdnn', 'lead12_rmssd', 'lead12_fft_ratio']

start = time.time()

def load_raw_data_on_demand_all_leads(filename, path):
    # Load the ECG data for all leads
    signal, _ = wfdb.rdsamp(path + filename)
    return signal  # Returns all leads as a 2D array

# Parallel processing for HRV feature extraction from all leads
def process_row_parallel_all_leads(idx, row, path):
    # Load signals for all leads
    signals = load_raw_data_on_demand_all_leads(row['filename_lr'] if sampling_rate == 100 else row['filename_hr'], path)
    
    results = {'idx': idx}  # Initialize results dictionary with the index
    
    # Process HRV features for each lead
    for lead_idx, lead_signal in enumerate(signals.T):  # Iterate over each lead
        # Extract HRV features for the current lead
        try:
            t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(lead_signal, sampling_rate=sampling_rate, show=False)[:3]

            # Add HRV metrics for the current lead
            rpeaks = t[rpeaks]
            results[f'lead{lead_idx+1}_sdnn'] = td.sdnn(rpeaks=rpeaks)["sdnn"]
            results[f'lead{lead_idx+1}_rmssd'] = td.rmssd(rpeaks=rpeaks)["rmssd"]
            results[f'lead{lead_idx+1}_fft_ratio'] = fd.welch_psd(rpeaks=rpeaks, show=False, show_param=False)["fft_ratio"]
        except:
            results[f'lead{lead_idx+1}_sdnn'] = np.nan
            results[f'lead{lead_idx+1}_rmssd'] = np.nan
            results[f'lead{lead_idx+1}_fft_ratio'] = np.nan

    return results

# Load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', skiprows = [i for i in range(1, 20_000) ], nrows=None, usecols=["age", "sex", "filename_lr", "filename_hr", "scp_codes"])
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Apply processing to each row using a generator for all leads
results = Parallel(n_jobs=-1)(
    delayed(process_row_parallel_all_leads)(idx, row, path)
    for idx, row in Y.iterrows()
)

# Convert results to a DataFrame
hrv_results_all_leads = pd.DataFrame(results).set_index('idx')

# Merge HRV results with the original DataFrame
Y = pd.concat([Y, hrv_results_all_leads], axis=1)

# Save or output the extended DataFrame
database_row = Y
print(database_row)

database_row.to_csv(f"ptbxl_database_with_hrv_time_features_{sampling_rate}_20_000-22_000.csv")

# Play a sound notification once all simulations are complete
if usewin:
    winsound.PlaySound("SystemQuestion", winsound.SND_ALIAS)

end = time.time()
print(f"It took {round(end - start)} seconds to initialize the model.")