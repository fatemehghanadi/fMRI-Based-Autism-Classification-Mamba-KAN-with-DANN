#%%
import os
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_craddock_2012
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
 
def fetch_and_load_abide1_data(data_directory, subject_ids):
    """
    Fetch the ABIDE I dataset preprocessed with the atlas and return time series data and valid subject IDs.
    Also, save valid subject IDs to a file.
    
    Parameters:
    - data_directory: Directory to save the data.
    - valid_id_file: Path to the file where valid subject IDs will be saved.

    Returns:
    - time_series_data: List of time series data for each subject.
    - valid_ids: List of valid subject IDs.
    """
    atlas = 'cc400'

    # Filt Global
    abide1_data = fetch_abide_pcp( 
        data_dir=data_directory,
        pipeline='cpac',
        band_pass_filtering=True,
        global_signal_regression=True,
        derivatives=f'rois_{atlas}',
        quality_checked=True
    )
    
    time_series_data = abide1_data[atlas]
    valid_ids = abide1_data['phenotypic']['FILE_ID'].tolist()

    # Just Extrac valid_ids
    print(f'len subject_ids : {len(subject_ids)}')
    valid_id_file = f'./home/Dataset/valid_subject_ids_NotQC_filtGlobal_{atlas}.txt'
    os.makedirs(os.path.dirname(valid_id_file), exist_ok=True)
    
    valid_ids = []
    time_series_data = []
    with open(valid_id_file, 'w') as f:
      
      for sid in subject_ids:

        print(f"{data_directory}/{sid}_rois_{atlas}.1D")
        subj=f'{sid}'
        if os.path.exists(f"{data_directory}/{subj}_rois_{atlas}.1D"):
            
            file_path = f"{data_directory}/{subj}_rois_{atlas}.1D"
            print(file_path)
            time_series = np.loadtxt(file_path)
            time_series_data.append(time_series)
            
            valid_ids.append(subj)

            f.write(f"{sid}\n")
            
            
    print(f' len time_series_data : {len(time_series_data)}')
    print(time_series_data[0].shape)
    print(f' len valid_ids : {len(valid_ids)}')
    return time_series_data, valid_ids

# def calculate_connectivity(time_series_data):
#     """
#     Calculate connectivity matrices using Pearson correlation.
    
#     Parameters:
#     - time_series_data: List of numpy arrays, each representing the time series of an individual subject.

#     Returns:
#     - connectivity_matrices: Numpy array of connectivity matrices for all subjects.
#     """
#     conn_measure = ConnectivityMeasure(kind='correlation')
#     connectivity_matrices = conn_measure.fit_transform(time_series_data)
    
#     # Optionally visualize one of the connectivity matrices
#     plt.imshow(connectivity_matrices[0], cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.title('Sample Connectivity Matrix')
#     plt.show()
    
#     return connectivity_matrices

def main(data_directory, csv_path, output_file, conn):
    """
    Main function to process the ABIDE I dataset and save the processed data.
    
    Parameters:
    - data_directory: Directory containing the downloaded .1D data files.
    - csv_path: Path to the phenotypic CSV file.
    - output_file: Path to the output file to save the processed data.
    - valid_id_file: Path to the file where valid subject IDs will be saved.
    """
    # Load phenotypic data
    phenotypic_data = pd.read_csv(csv_path)
    subject_ids = phenotypic_data['FILE_ID'].tolist()
    print(f'len subject_ids : {len(subject_ids)}')
    # Fetch and load ABIDE I data
    time_series_data, valid_ids = fetch_and_load_abide1_data(data_directory, subject_ids)

    # Calculate connectivity matrices
    if conn == 'PCC':
        print('pcc')
        # connectivity_matrices = calculate_connectivity(time_series_data)
        conn_measure = ConnectivityMeasure(kind='correlation')
        connectivity_matrices = conn_measure.fit_transform(time_series_data)
        
    elif conn == 'TPE':
        vectorize = False
        conn_measure = ConnectivityMeasure(kind='correlation')
        input_data = conn_measure.fit_transform(time_series_data)
        conn_measure = ConnectivityMeasure(kind='tangent', vectorize=vectorize, discard_diagonal=False)
        conn_measure.fit(input_data)

        conn_vec = conn_measure.transform(input_data)
        
        connectivity_matrices=conn_vec      
        
    elif conn == 'SPR':
        print('spr calc')
        # Calculate Spearman's correlation coefficients between ROIs
        num_datas = len(time_series_data)
        num_rois = 200
        print (f'num data : {num_datas}')
        connectivity_matrices = np.zeros((num_datas,num_rois, num_rois))
        n=0
        for D in time_series_data:
            corr = spearmanr(D)
            print(corr[0])
            print(corr[0].shape)
            connectivity_matrices[n] = corr[0]
            print(f'n : {n}')
        #     for i in range(num_rois):
        #         for j in range(num_rois):
        #             if i != j:  # Avoid calculating correlation of a ROI with itself
        #                 corr, _ = spearmanr(D[:,i], D[:,j])
        #                 connectivity_matrices[n, i, j] = corr
            n = n+1

    elif conn == 'null':
        
        # Find the maximum length of the time series
        max_length = max(data.shape[0] for data in time_series_data)
        print(f'max_length:{max_length}')
        # Pad each time series with zeros
        padded_time_series_data = []
        for data in time_series_data:
            padded_data = np.zeros((data.shape[1], max_length))
            padded_data[:, :data.shape[0]] = data.T
            padded_time_series_data.append(padded_data)

        # Convert the list to a NumPy array
        connectivity_matrices = np.array(padded_time_series_data)

        # Print shapes to verify
        print([data.shape for data in connectivity_matrices])

        




    # Filter phenotypic data for valid subjects
    valid_phenotypic_data = phenotypic_data[phenotypic_data['FILE_ID'].isin(valid_ids)]
    
    # Extract subject IDs and labels for the valid subjects
    subject_ids = valid_phenotypic_data['FILE_ID'].tolist()
    labels = valid_phenotypic_data['DX_GROUP'].tolist()
    age = valid_phenotypic_data['AGE_AT_SCAN'].tolist()
    gender = valid_phenotypic_data['SEX'].tolist()
    handedness = valid_phenotypic_data['HANDEDNESS_CATEGORY'].tolist()
    fiq = valid_phenotypic_data['FIQ'].tolist()
    viq = valid_phenotypic_data['VIQ'].tolist()
    piq = valid_phenotypic_data['PIQ'].tolist()
    eye = valid_phenotypic_data['EYE_STATUS_AT_SCAN'].tolist()
    site = valid_phenotypic_data['SITE_ID'].tolist()
    
    print(f'-------- dataMatrix.shape: {connectivity_matrices.shape}')
    print(f'-------- subject_ids.shape: {len(subject_ids)}')
    print(f'-------- labels.shape: {len(labels)}')
    print(f'-------- age.shape: {len(age)}')
    print(f'-------- gender.shape: {len(gender)}')
    print(f'-------- handedness.shape: {len(handedness)}')
    print(f'-------- fiq.shape: {len(fiq)}')
    print(f'-------- viq.shape: {len(viq)}')
    print(f'-------- piq.shape: {len(piq)}')
    print(f'-------- eye.shape: {len(eye)}')
    print(f'-------- site.shape: {len(site)}')

    # Shuffle data
    connectivity_matrices, labels, subject_ids, age, gender, handedness, fiq, viq, piq, eye, site = shuffle(
        connectivity_matrices, labels, subject_ids, age, gender, handedness, fiq, viq, piq, eye,site, random_state=1234
    )
    
    # Save processed data
    np.savez(output_file, fc=connectivity_matrices, label=labels, subject=subject_ids, age=age, gender=gender, handedness=handedness, fiq=fiq, viq=viq, piq=piq, eye=eye, site=site)

def create_fold_files(data_pth, num_folds):
    """
    Create fold files from the processed data for cross-validation.
    
    Parameters:
    - data_pth: Path to the processed data file.
    - num_folds: Number of folds for cross-validation.
    """
    data = np.load(data_pth, allow_pickle=True)
    fc = data['fc']
    labels = data['label']
    subject_ids = data['subject']
    
    fold_size = len(fc) // num_folds
    
    for fold_idx in range(num_folds):
        start_idx = fold_idx * fold_size
        end_idx = (fold_idx + 1) * fold_size if fold_idx < num_folds - 1 else len(subject_ids)
        trn_idx = np.concatenate((np.arange(0, start_idx), np.arange(end_idx, len(subject_ids))))
        val_idx = np.arange(start_idx, end_idx)
        tst_idx = np.arange(start_idx, end_idx)
        
        print(f'Fold {fold_idx + 1}: Train indices ~ {start_idx}-{end_idx}, Validation indices {start_idx}-{end_idx}, Test indices {start_idx}-{end_idx}')
        fold_dir = f"./home/Dataset/CV_5_/rp1_f{fold_idx + 1}.npz"
        
        os.makedirs(os.path.dirname(fold_dir), exist_ok=True)
        np.savez(fold_dir, trn_idx=trn_idx, val_idx=val_idx, tst_idx=tst_idx)


# Example usage
# data_directory = './temp/ABIDE_pcp/cpac/filt_noglobal'
# csv_path = './temp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
# output_file = './home/Dataset/ABIDE_CC200_PCC_withSITE_shuffle_randomSeed1234_.npz'

# data_directory = './temp/ABIDE_pcp/cpac/global/ABIDE_pcp/cpac/filt_global_NotQC/ABIDE_pcp/cpac/filt_global'
data_directory ='./temp/ABIDE_pcp/cpac/global/ABIDE_pcp/cpac/filt_global'
csv_path = './temp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
output_file = './home/Dataset/ABIDE_CC400_NotQC_PCC_withSITE_shuffle_randomSeed1234_filtGlobal.npz'

main(data_directory, csv_path, output_file, conn='PCC')
# create_fold_files(data_pth=output_file, num_folds=5)

# ABIDE_CC400_NOTQC_TPE_withSITE_shuffle_randomSeed1234_filtGlobal

# %%
