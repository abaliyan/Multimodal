"""Multi-modal Machine Learning Pipeline for Specific Activity (SA) Prediction.

This script implements a comprehensive ML pipeline specifically for predicting
Specific Activity (SA) from multi-modal spectroscopic data including XANES,
EXAFS, XRD, PDF, SAXS, and HAXPES measurements.
"""

# Core scientific computing libraries
from sklearn.metrics import r2_score
import os
import itertools
import numpy as np
import pandas as pd
import pickle
import random
import csv
from matplotlib import pyplot as plt

# Machine learning models and utilities
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics.pairwise import rbf_kernel
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from itertools import permutations

# Data processing and interpolation
from scipy.interpolate import interp1d
import scipy.interpolate
import datetime
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================\n# CONFIGURATION AND INITIALIZATION\n# =============================================================================\n\n# Output directory for model artifacts and results\nDIR_Data = "models"

def find_missing_files(list1, list2):
    """Find files present in list1 but not in list2."""
    return list(set(list1) - set(list2))

def getTime():
    """Get current date timestamp for file naming."""
    return datetime.date.today().strftime("%Y-%m-%d")

def normalize_data(data):
    """Normalize data to range [0, 1] using min-max scaling."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Function to read the csv file
def ReadCsvData(filename):
    """Read and preprocess spectroscopic data from Excel files for SA prediction.

    This function handles different spectroscopic techniques with technique-specific
    data range selections and preprocessing steps optimized for Specific Activity prediction.

    Args:
        filename (str): Path to Excel file containing spectroscopic data

    Returns:
        list: Two-element list [x_values, y_values] or [[]] if file not found

    Spectroscopic Techniques Handled:
        - XRD: X-ray diffraction patterns (trimmed edges for noise reduction)
        - XANES: X-ray absorption near edge structure (excludes post-edge region)
        - EXAFS: Extended X-ray absorption fine structure (k-space, limited to 281 points)
        - PDF: Pair distribution function (excludes low-r region artifacts)
        - SAXS: Small angle X-ray scattering (full range)
        - HAXPES_VB: Valence band photoelectron spectroscopy (excludes secondary electrons)
        - HAXPES_Pt4f/Pt3d: Core-level photoelectron spectroscopy (Pt 4f and 3d regions)
    """
    try:
        Localfiledata = []
        if "XRD" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            Localfiledata = [df[0].tolist()[50:-50], df[1].tolist()[50:-50]]
        elif "XANES" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            if (len(df[0]) and len(df[1])) != 0:
                Localfiledata = [df[0].tolist()[0:-86], df[1].tolist()[0:-86]]
        elif "EXAFS" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            # Localfiledata = [df[0].tolist()[0:281], df[1].tolist()[0:281]]
            Localfiledata = [df[0].tolist()[0:281], df[1].tolist()[0:281]]
        elif "PDF" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            if (len(df[0]) and len(df[1])) != 0:
                Localfiledata = [df[0].tolist()[50:], df[1].tolist()[50:]]
        elif "SAXS" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            if (len(df[0]) and len(df[1])) != 0:
                Localfiledata = [df[0].tolist()[0:], df[1].tolist()[0:]]
        elif "HAXPES_VB" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            if (len(df[0]) and len(df[1])) != 0:
                Localfiledata = [df[0].tolist()[77:], df[1].tolist()[77:]]
        elif "HAXPES_Pt4f" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            if (len(df[0]) and len(df[1])) != 0:
                Localfiledata = [df[0].tolist(), df[1].tolist()]
        elif "HAXPES_Pt3d" in filename:
            df = pd.read_excel(filename, skiprows=None, header=None)
            if (len(df[0]) and len(df[1])) != 0:
                Localfiledata = [df[0].tolist(), df[1].tolist()]
    except FileNotFoundError:
        print('File not found')
        pass
    if len(Localfiledata) != 0:
        return Localfiledata
    else:
        return [[]]

# Fuction to plot the figures
def PlotCustumFigure(data, foldername='', Data_type='', XaxisFlag=True, xdata=np.empty(0), RangeStart=0, RangeEnd=0, NormFlag=True):
    if len(data) != 0 and np.isnan(data).any() == False:
        fig = plt.figure(1, figsize=(5, 4))
        # Code to shorten the data range [Min, max]
        data = data[RangeStart:RangeEnd]

        if NormFlag == True:
            data = normalize_data(data)
        if XaxisFlag == True:
            x = xdata
            xMin = np.min(x)
            xMax = np.max(x)
        else:
            x = []
        yMin = np.min(data)
        yMax = np.max(data)

        if len(x) != 0:
            plt.plot(x, data)
            plt.axis([xMin, xMax, (yMin+yMin*0.05), (yMax+yMax*0.05)])
            if Data_type != '':
                # custumtitle = foldername+  '_'+ Data_type
                custumtitle = Data_type
                if custumtitle.startswith('_'):
                    custumtitle = custumtitle[1:]
                plt.title(custumtitle)
                save_figure_to_folder(fig, foldername, Data_type)
            # plt.show()
            if plt.fignum_exists(fig.number):
                # close the figure
                plt.close(fig.number)
        else:
            plt.plot(data)
            if Data_type != '':
                # custumtitle = foldername+  '_'+ Data_type
                custumtitle = Data_type
                if custumtitle.startswith('_'):
                    custumtitle = custumtitle[1:]
                plt.title(custumtitle)
                save_figure_to_folder(fig, foldername, Data_type)
            # plt.show()
            if plt.fignum_exists(fig.number):
                # close the figure
                plt.close(fig.number)
    else:
        print('Data was empty')
        fig = plt.figure(1, figsize=(5, 4))
        plt.text(0.5, 0.5, 'No data available',
                 ha='center', va='center', fontsize=20)
        plt.axis('off')
        if Data_type != '':
            # custumtitle = foldername+  '_'+ Data_type
            custumtitle = Data_type
            if custumtitle.startswith('_'):
                custumtitle = custumtitle[1:]
        plt.title(custumtitle)
        save_figure_to_folder(fig, foldername, Data_type)
        if plt.fignum_exists(fig.number):
            # close the figure
            plt.close(fig.number)

# Fuction to plot the figures
def PlotCustumFigureChar(datadict, dictkey):
    foldername = 'Analysis/ViwAll'
    Data_type = dictkey
    NormFlag = False
    fig = plt.figure(1, figsize=(5, 4))
    for dataindex, datakey in enumerate(datadict[dictkey]):
        ydata = datadict[dictkey][datakey][1]
        xdata = datadict[dictkey][datakey][0]
        if len(ydata) != 0:
            if NormFlag == True:
                ydata = normalize_data(ydata)
                plt.plot(xdata, ydata, label=datakey)
                # plt.show()
            else:
                plt.plot(xdata, ydata, label=datakey)
    plt.title(dictkey)
    plt.legend()
    # plt.show()
    save_figure_to_folder(fig, foldername, Data_type)
    if plt.fignum_exists(fig.number):
        # close the figure
        plt.close(fig.number)

# Fuction to plot the figures
def PlotCustumFigureMulti(data, foldername='', Data_type='', XaxisFlag=True, RangeStart=0, RangeEnd=0, NormFlag=True):
    if len(data) != 0:
        fig = plt.figure(1, figsize=(5, 4))
        # Code to shorten the data range [Min, max]
        data = data[RangeStart:RangeEnd]

        if NormFlag == True:
            data = normalize_data(data)
        if XaxisFlag == True:
            x = np.arange(0, data.size, 1)
            xMin = np.min(x)
            xMax = np.max(x)
        else:
            x = []
        yMin = np.min(data)
        yMax = np.max(data)

        if len(x) != 0:
            plt.plot(x, data)
            plt.axis([xMin, xMax, (yMin+yMin*0.05), (yMax+yMax*0.05)])
            if Data_type != '':
                # custumtitle = foldername+  '_'+ Data_type
                custumtitle = Data_type
                if custumtitle.startswith('_'):
                    custumtitle = custumtitle[1:]
                plt.title(custumtitle)
                save_figure_to_folder(fig, foldername, Data_type)
            # plt.show()
            if plt.fignum_exists(fig.number):
                # close the figure
                plt.close(fig.number)
        else:
            plt.plot(data)
            if Data_type != '':
                # custumtitle = foldername+  '_'+ Data_type
                custumtitle = Data_type
                if custumtitle.startswith('_'):
                    custumtitle = custumtitle[1:]
                plt.title(custumtitle)
                save_figure_to_folderMulti(fig, foldername, Data_type)
            # plt.show()
            if plt.fignum_exists(fig.number):
                # close the figure
                plt.close(fig.number)
    else:
        print('Data was empty')

# Fuction to save the figures
def save_figure_to_folder(figure, folder_name='', file_name=''):
    # define the name of the directory to be created
    if folder_name != '':
        localresultpath = f"{DIR_Data}" + '/' + folder_name
        localresultpath = localresultpath + '_' + getTime()
    if folder_name == 'SingleDataSet':
        localresultpath = f"{DIR_Data}/Analysis/SingleDataSet/"
    if folder_name == 'DoubleDataSet':
        localresultpath = f"{DIR_Data}/Analysis/DoubleDataSet/"
    if folder_name == 'TripleDataSet':
        localresultpath = f"{DIR_Data}/Analysis/TripleDataSet/"
    if folder_name == 'QuadDataSet':
        localresultpath = f"{DIR_Data}/Analysis/QuadDataSet/"
    if folder_name == 'PentaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/PentaDataSet/"
    if folder_name == 'HexaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/HexaDataSet/"
    if folder_name == 'HeptaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/HeptaDataSet/"
    if folder_name == 'OctaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/OctaDataSet/"

    try:
        # os.makedirs(path)
        if not os.path.exists(localresultpath):
            os.makedirs(localresultpath)
    except OSError:
        print(f"Creation of the directory {localresultpath} failed")
    else:
        print(f"Successfully created the directory {localresultpath}")

    file_name = file_name + '.png'
    file_path = os.path.join(localresultpath, file_name)
    figure.savefig(file_path)

# Fuction to save the figures
def save_figure_to_folderMulti(figure, folder_name='', file_name=''):
    # define the name of the directory to be created
    if folder_name != '':
        localresultpath = f"{DIR_Data}" + '/' + folder_name
    if folder_name == 'SingleDataSet':
        localresultpath = f"{DIR_Data}/Analysis/SingleDataSet/"
    if folder_name == 'DoubleDataSet':
        localresultpath = f"{DIR_Data}/Analysis/DoubleDataSet/"
    if folder_name == 'TripleDataSet':
        localresultpath = f"{DIR_Data}/Analysis/TripleDataSet/"
    if folder_name == 'QuadDataSet':
        localresultpath = f"{DIR_Data}/Analysis/QuadDataSet/"
    if folder_name == 'PentaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/PentaDataSet/"
    if folder_name == 'HexaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/HexaDataSet/"
    if folder_name == 'HeptaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/HeptaDataSet/"
    if folder_name == 'OctaDataSet':
        localresultpath = f"{DIR_Data}/Analysis/OctaDataSet/"

    try:
        # os.makedirs(path)
        if not os.path.exists(localresultpath):
            os.makedirs(localresultpath)
    except OSError:
        print(f"Creation of the directory {localresultpath} failed")
    else:
        print(f"Successfully created the directory {localresultpath}")

    file_name = file_name + '.png'
    file_path = os.path.join(localresultpath, file_name)
    figure.savefig(file_path)

# Function to convert from list o array
def convert_to_array(var):
    if isinstance(var, list):
        return np.array(var)
    else:
        return var

# Function to find the maximum index length
def max_index(lst):
    max_val = max(lst)
    max_index = lst.index(max_val)
    return max_val, max_index

# Function to find the minimum index length
def min_index(lst):
    min_val = min(lst)
    min_index = lst.index(min_val)
    return min_val, min_index

# Fuction to the Extract the default dataset
def getDefaultData(DatatypeName, DataDictionary, DataNameLitral, plotIndividualFlag, datatype):
    counter = 0
    for datakeys in DataDictionary.keys():
        if DataDictionary[datakeys][datatype] != [[]]:
            if isinstance(DataDictionary[datakeys][datatype], list):
                DatatypeName['y' + str(DataNameLitral[counter])] = convert_to_array(
                    DataDictionary[datakeys][datatype][1]).flatten()
        else:
            DatatypeName['y' + str(DataNameLitral[counter])
                         ] = convert_to_array(DataDictionary[datakeys][datatype][0])
        counter = counter+1

    # check the minimum and maximum of the list of arrays for interpolate/extrapolate
    rows = [len(row) for keys, row in DatatypeName.items() if len(row) != 0]
    mykeys = [keys for keys, row in DatatypeName.items() if len(row) != 0]
    maxElement, maxindex = max_index(rows)
    minElement, minindex = min_index(rows)
    mykeys_max = mykeys[maxindex]
    mykeys_min = mykeys[minindex]

    counter = 0
    for datakeys in DataDictionary.keys():
        if DataDictionary[datakeys][datatype] != [[]]:
            if isinstance(DataDictionary[datakeys][datatype], list):
                DatatypeName['y' + str(DataNameLitral[counter])] = (convert_to_array(DataDictionary[datakeys][datatype][0]).flatten(),
                                                                    convert_to_array(DataDictionary[datakeys][datatype][1]).flatten())
        else:
            DatatypeName['y' + str(DataNameLitral[counter])
                         ] = convert_to_array(DataDictionary[datakeys][datatype][0])
        counter = counter+1

    return minElement, maxElement, DatatypeName, mykeys_max, mykeys_min

# Function to Interpolate Data
def interpolatedata(old_x, old_y, new_x, type='linear'):
    f = scipy.interpolate.interp1d(
        old_x, old_y, kind=type, fill_value='extrapolate')
    result = f(new_x)
    return result, new_x

# Function to extrapolate Data
def extrapolatedata(old_x, old_y, new_x, type=1):
    f = interp1d(old_x, old_y, type, fill_value='extrapolate')
    new_x = np.linspace(new_x[0], new_x[-1], len(new_x))
    result = f(new_x)
    return result, new_x

# Function to extrapolate Data
def fixedlendata(old_x, old_y, new_x, type=1):
    f = interp1d(old_x, old_y, type, fill_value='extrapolate')
    new_x = np.linspace(old_x[0], old_x[-1], len(new_x))
    result = f(new_x)
    return result, new_x

# Fuction to the Intrapolate/Extraplotate the dataset
def getInterXtrapolatedata(Data_old, flag, NormFlag, minElement, maxElement, key_max, key_min):
    Data_New = {}
    Data_New_XY = {}
    if flag == 'Xtra':
        for keys, data in Data_old.items():
            if len(data) == 0:
                Data_New[keys] = data
                Data_New_XY[keys] = [data, data]
            else:
                extrapoloteddata, new_x = extrapolatedata(
                    data[0], data[1], Data_old[key_max][0], type=1)
                if NormFlag == False:
                    Data_New[keys] = extrapoloteddata
                    Data_New_XY[keys] = [new_x, extrapoloteddata]
                if NormFlag == True:
                    # Data_New[keys] = normalize_data(data = extrapoloteddata.tolist())
                    Data_New[keys] = normalize_data(extrapoloteddata)
                    Data_New_XY[keys] = [
                        new_x, normalize_data(extrapoloteddata)]
    if flag == 'Inter':
        for keys, data in Data_old.items():
            if len(data) == 0:
                Data_New[keys] = data
                Data_New_XY[keys] = [data, data]
            else:
                print(keys)
                # interpolatedata(data[0], data[1], Data_old[key_min][0], type = 'linear')
                interpolateddata, new_x = interpolatedata(
                    data[0], data[1], Data_old[key_min][0], type='linear')
                if NormFlag == False:
                    Data_New[keys] = interpolateddata
                    Data_New_XY[keys] = [new_x, interpolateddata]
                if NormFlag == True:
                    # Data_New[keys] = normalize_data(data = interpolateddata.tolist())
                    Data_New[keys] = normalize_data(interpolateddata)
                    Data_New_XY[keys] = [
                        new_x, normalize_data(interpolateddata)]
    if flag == 'fixed':
        desiredlength = 200
        for keys, data in Data_old.items():
            if len(data) == 0:
                Data_New[keys] = data
                Data_New_XY[keys] = [data, data]
            else:
                regrid = np.linspace(
                    0, data[0].size, desiredlength).astype(float)
                resample, new_x = fixedlendata(
                    data[0], data[1], regrid, type=1)
                if NormFlag == False:
                    Data_New[keys] = resample
                    Data_New_XY[keys] = [new_x, resample]
                if NormFlag == True:
                    # Data_New[keys] = normalize_data(data =  resample.tolist())
                    Data_New[keys] = normalize_data(resample)
                    Data_New_XY[keys] = [new_x, normalize_data(resample)]
    return Data_New, Data_New_XY


# Fuction to create the ML models objects
def listofMLmodels(models):
    MLmodellist = []
    for model in models:
        MLmodellist.append(model)
    return MLmodellist

# Fuction to Concatinate the Dataset horigentially
def customConcatinate(datalist):
    result = datalist[0]
    for i in range(1, len(datalist)):
        result = np.concatenate((result, datalist[i]), axis=1)
    return result

# Fuction to the Prapare the xdataset and ydataset
def prepareXandYdata(listofdata_x, DataDictionary, listofdata_y):
    if len(listofdata_x) != len(listofdata_y):
        print('The X and Y did not have same element')
        return
    else:
        xdata = []
        ydata = []
        for arrx in listofdata_x:
            xdata.append(DataDictionary[arrx])
        for arry in listofdata_y:
            ydata.append(arry)
        return np.array(xdata), np.array(ydata)

# Fuction to the train results
def plotTrainResult(train, test, gtest, legendtxt, foldername='', Data_type=''):

    if train.ndim == 1:
        fig = plt.figure(figsize=(5, 5))
        if len(train) != 0:
            plt.plot(train, 'ro', label="Train")
        if len(test) != 0:
            plt.plot(test, 'bs', label='Test')
        if len(gtest) != 0:
            plt.plot(gtest, 'gD', label='Gen1')
        plt.legend()
        plt.xlabel('Train/Test Sample')
        plt.ylabel('SA (A/m2-Pt)')
        plt.title(legendtxt)
        plt.grid(True)

    else:
        # Create a figure and a grid of subplots with a single call
        fig = plt.figure(figsize=(9, 3))
        gs = fig.add_gridspec(1, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot([i[0] for i in train], 'ro', label="Train")
        ax1.plot([i[0] for i in test], 'bs', label='Test')
        ax1.set_title('ESCA')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot([i[1] for i in train], 'ro', label="Train")
        ax2.plot([i[1] for i in test], 'bs', label='Test')
        ax2.set_title('SA')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot([i[2] for i in train], 'ro', label="Train")
        ax3.plot([i[2] for i in test], 'bs', label='Test')
        ax3.set_title('MA')

        fig.suptitle(legendtxt)
    if Data_type != '':
        save_figure_to_folder(fig, foldername, Data_type+'_Model')
        # plt.show()
    if plt.fignum_exists(fig.number):
        # close the figure
        plt.close(fig.number)
    # plt.show()
    return


def checkPermutationsnew(models, DataArray, x_lst, y_lst, training_seed=334, num_evaluation_seeds=1):
    modalities = list(DataArray.keys())
    groups = list(range(len(x_lst)))
    best_models = {}

    # Directory to save log files
    log_dir = f"{DIR_Data}/model_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Log files
    training_log_file = os.path.join(
        log_dir, f"Training_log_seed_{training_seed}.txt")
    evaluation_log_file = os.path.join(
        log_dir, f"Evaluation_log_{num_evaluation_seeds}_seeds.txt")

    print("="*60)
    print("PHASE 1: Training and Saving Models for Seed", training_seed)
    print("="*60)

    # Initialize training log file
    with open(training_log_file, "w") as log:
        log.write(f"Model Training for Seed {training_seed}\n")
        log.write(f"Started: {datetime.datetime.now()}\n")
        log.write("="*50 + "\n\n")

    # PHASE 1: Train and save models for training_seed using original Training_Testing_models function
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.08, random_state=training_seed)
    train_idx, test_idx = next(splitter.split(x_lst, y_lst, groups=groups))

    print(
        f"Training seed {training_seed}: Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # CHANGED: Only store file paths and metadata, not actual models
    saved_models_info = {}

    # Loop through different combination sizes for training
    for pair in range(len(modalities), 0, -1):  # Start from 8 for OctaDataSet
        perm_list = list(permutations(modalities, pair))
        print(
            f'Training models for {pair}-combinations: {len(perm_list)} permutations')

        # Determine fold name based on combination size
        if pair == 1:
            foldName_base = 'SingleDataSet'
        elif pair == 2:
            foldName_base = 'DoubleDataSet'
        elif pair == 3:
            foldName_base = 'TripleDataSet'
        elif pair == 4:
            foldName_base = 'QuadDataSet'
        elif pair == 5:
            foldName_base = 'PentaDataSet'
        elif pair == 6:
            foldName_base = 'HexaDataSet'
        elif pair == 7:
            foldName_base = 'HeptaDataSet'
        elif pair == 8:
            foldName_base = 'OctaDataSet'
        else:
            foldName_base = f'{pair}DataSet'

        # Prepare combined training data for this combination size
        all_X_train = []
        all_Y_train = []
        all_groups_train = []

        # Process each permutation and combine into single dataset
        for perm_index, perm in enumerate(perm_list):
            def build_features(sample_idx):
                x_sample = x_lst[sample_idx]
                feat_list = []
                for mod in perm:
                    X_mod, _ = prepareXandYdata(
                        [x_sample], DataArray[mod], [y_lst[sample_idx]])
                    if len(X_mod) > 0:
                        feat_list.append(X_mod[0])
                    else:
                        return None
                return np.concatenate(feat_list)

            for i in train_idx:
                feat = build_features(i)
                if feat is not None:
                    all_X_train.append(feat)
                    all_Y_train.append(y_lst[i])
                    all_groups_train.append(i + perm_index * len(groups))

        # Prepare test data for this combination size (needed for Training_Testing_models)
        all_X_test = []
        all_Y_test = []
        all_groups_test = []

        # Process each permutation and combine into single dataset for test
        for perm_index, perm in enumerate(perm_list):
            def build_features(sample_idx):
                x_sample = x_lst[sample_idx]
                feat_list = []
                for mod in perm:
                    X_mod, _ = prepareXandYdata(
                        [x_sample], DataArray[mod], [y_lst[sample_idx]])
                    if len(X_mod) > 0:
                        feat_list.append(X_mod[0])
                    else:
                        return None
                return np.concatenate(feat_list)

            for i in test_idx:
                feat = build_features(i)
                if feat is not None:
                    all_X_test.append(feat)
                    all_Y_test.append(y_lst[i])
                    all_groups_test.append(i + perm_index * len(groups))

        # Use original Training_Testing_models function
        foldName = f'{foldName_base}_Seed{training_seed}'
        trained_models_dict = Training_Testing_models(
            models,
            np.vstack(all_X_train),
            np.array(all_Y_train),
            all_groups_train,
            np.vstack(all_X_test),
            np.array(all_Y_test),
            all_groups_test,
            foldName,
            training_log_file  # Pass log file to the training function
        )

        # CHANGED: Store only metadata, not actual models
        trained_models_for_fold = []
        output_filename = f"{DIR_Data}/models_scores_SA/"

        for model_idx, original_model in enumerate(models):
            model_filename = f"{output_filename}{foldName}_model_{model_idx+1}_{original_model.__class__.__name__}_SA.pkl"

            trained_models_for_fold.append({
                'model_name': original_model.__class__.__name__,
                'model_path': model_filename,
                'model_index': model_idx + 1
            })

        saved_models_info[foldName] = trained_models_for_fold

    # Save split indices for reference
    split_info = {
        'train_idx': train_idx.tolist(),
        'test_idx': test_idx.tolist(),
        'seed': training_seed
    }
    split_file = os.path.join(
        log_dir, f"split_indices_seed_{training_seed}.pkl")
    with open(split_file, 'wb') as f:
        pickle.dump(split_info, f)

    with open(training_log_file, "a") as log:
        log.write(
            f"\nAll models trained and saved: {datetime.datetime.now()}\n")
        log.write("="*50 + "\n")

    print(f"Phase 1 completed. Models saved using Training_Testing_models function.")

    print("\n" + "="*60)
    num_evaluation_seeds = [166, 214, 220, 516, 379, 747, 809, 203, 204, 407]
    print(
        f"PHASE 2: Cross-Validation Training on {len(num_evaluation_seeds)} Different Seeds")
    print("="*60)

    # Initialize evaluation log file
    with open(evaluation_log_file, "w") as log:
        log.write(
            f"Cross-Validation Training on {len(num_evaluation_seeds)} Different Seeds\n")
        log.write(f"Purpose: Show models are not overfitting\n")
        log.write(f"Started: {datetime.datetime.now()}\n")
        log.write("="*50 + "\n\n")

    # PHASE 2: Train models on multiple seeds for cross-validation (to show no overfitting)
    all_cv_results = []

    # Loop through different combination sizes for cross-validation
    for pair in range(len(modalities), 0, -1):  # Start from 8 for OctaDataSet
        perm_list = list(permutations(modalities, pair))

        # Determine fold name based on combination size
        if pair == 1:
            foldName_base = 'SingleDataSet'
        elif pair == 2:
            foldName_base = 'DoubleDataSet'
        elif pair == 3:
            foldName_base = 'TripleDataSet'
        elif pair == 4:
            foldName_base = 'QuadDataSet'
        elif pair == 5:
            foldName_base = 'PentaDataSet'
        elif pair == 6:
            foldName_base = 'HexaDataSet'
        elif pair == 7:
            foldName_base = 'HeptaDataSet'
        elif pair == 8:
            foldName_base = 'OctaDataSet'
        else:
            foldName_base = f'{pair}DataSet'

        print(
            f'Cross-validation training for {pair}-combinations: {len(perm_list)} permutations across {len(num_evaluation_seeds)} seeds')

        # Train models across multiple seeds (cross-validation)
        for cv_seed in num_evaluation_seeds:
            splitter_cv = GroupShuffleSplit(
                n_splits=1, test_size=0.08, random_state=cv_seed)
            train_idx_cv, test_idx_cv = next(
                splitter_cv.split(x_lst, y_lst, groups=groups))

            # Prepare training data for this CV seed
            all_X_train_cv = []
            all_Y_train_cv = []
            all_groups_train_cv = []

            # Prepare test data for this CV seed
            all_X_test_cv = []
            all_Y_test_cv = []
            all_groups_test_cv = []

            # Process each permutation and combine into single dataset
            for perm_index, perm in enumerate(perm_list):
                def build_features(sample_idx):
                    x_sample = x_lst[sample_idx]
                    feat_list = []
                    for mod in perm:
                        X_mod, _ = prepareXandYdata(
                            [x_sample], DataArray[mod], [y_lst[sample_idx]])
                        if len(X_mod) > 0:
                            feat_list.append(X_mod[0])
                        else:
                            return None
                    return np.concatenate(feat_list)

                # Build training features
                for i in train_idx_cv:
                    feat = build_features(i)
                    if feat is not None:
                        all_X_train_cv.append(feat)
                        all_Y_train_cv.append(y_lst[i])
                        all_groups_train_cv.append(
                            i + perm_index * len(groups))

                # Build test features
                for i in test_idx_cv:
                    feat = build_features(i)
                    if feat is not None:
                        all_X_test_cv.append(feat)
                        all_Y_test_cv.append(y_lst[i])
                        all_groups_test_cv.append(i + perm_index * len(groups))

            if len(all_X_train_cv) == 0 or len(all_X_test_cv) == 0:
                continue

            # Use original Training_Testing_models function for cross-validation training
            cv_foldName = f'{foldName_base}_CV_Seed{cv_seed}'
            cv_trained_models_dict = Training_Testing_models(
                models,
                np.vstack(all_X_train_cv),
                np.array(all_Y_train_cv),
                all_groups_train_cv,
                np.vstack(all_X_test_cv),
                np.array(all_Y_test_cv),
                all_groups_test_cv,
                cv_foldName,
                evaluation_log_file  # Pass log file to the training function
            )

            # Load the results from the Excel file created by Training_Testing_models
            output_filename = f"{DIR_Data}/models_scores_SA/"
            results_file = f"{output_filename}{cv_foldName}_SA_model_mse_scores.xlsx"

            if os.path.exists(results_file):
                cv_results_df = pd.read_excel(results_file)
                cv_results_df['Combination_Size'] = pair
                cv_results_df['Fold_Name'] = foldName_base
                cv_results_df['CV_Seed'] = cv_seed
                all_cv_results.append(cv_results_df)


        with open(evaluation_log_file, "a") as log:
            log.write(
                f"Completed cross-validation training for {foldName_base}\n")

    # Combine all CV results and update Phase 1 Excel files with mean CV scores
    if all_cv_results:
        df_all_cv_results = pd.concat(all_cv_results, ignore_index=True)

        # Calculate mean CV R2 Test scores for each model and combination
        mean_cv_scores = df_all_cv_results.groupby(
            ['Combination_Size', 'Fold_Name', 'Model'])['R2 Test'].mean().round(4)

        # Update Phase 1 Excel files with mean CV scores
        print("Updating Phase 1 Excel files with mean CV R² scores...")

        for pair in range(len(modalities), 0, -1):
            # Determine fold name
            if pair == 1:
                foldName_base = 'SingleDataSet'
            elif pair == 2:
                foldName_base = 'DoubleDataSet'
            elif pair == 3:
                foldName_base = 'TripleDataSet'
            elif pair == 4:
                foldName_base = 'QuadDataSet'
            elif pair == 5:
                foldName_base = 'PentaDataSet'
            elif pair == 6:
                foldName_base = 'HexaDataSet'
            elif pair == 7:
                foldName_base = 'HeptaDataSet'
            elif pair == 8:
                foldName_base = 'OctaDataSet'
            else:
                foldName_base = f'{pair}DataSet'

            # Phase 1 Excel file path
            phase1_foldName = f'{foldName_base}_Seed{training_seed}'
            output_filename = f"{DIR_Data}/models_scores_SA/"
            phase1_excel_file = f"{output_filename}{phase1_foldName}_SA_model_mse_scores.xlsx"

            if os.path.exists(phase1_excel_file):
                # Read Phase 1 Excel file
                phase1_df = pd.read_excel(phase1_excel_file)

                # Add Mean_CV_R2_Test column
                cv_scores_for_combination = []
                for _, row in phase1_df.iterrows():
                    model_name = row['Model']
                    try:
                        cv_score = mean_cv_scores[(
                            pair, foldName_base, model_name)]
                        cv_scores_for_combination.append(cv_score)
                    except KeyError:
                        # If no CV score found for this model, use NaN
                        cv_scores_for_combination.append(np.nan)

                phase1_df['Mean_CV_R2_Test'] = cv_scores_for_combination

                # Save updated Excel file
                phase1_df.to_excel(phase1_excel_file, index=False)

                with open(evaluation_log_file, "a") as log:
                    log.write(
                        f"Updated {phase1_excel_file} with mean CV R2 scores\n")

        # Calculate overall summary statistics
        summary_stats = df_all_cv_results.groupby(['Combination_Size', 'Fold_Name', 'Model']).agg({
            'R2 Train': ['mean', 'std'],
            'R2 Test': ['mean', 'std'],
            'MSE Train': ['mean', 'std'],
            'MSE Test': ['mean', 'std'],
            'MAE Train': ['mean', 'std'],
            'MAE Test': ['mean', 'std'],
            'Explained Variance': ['mean', 'std'],
            'Pearson R': ['mean', 'std'],
            'Spearman Rho': ['mean', 'std'],
            'Kendall Tau': ['mean', 'std']
        }).round(4)

        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip()
                                 for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()

        # Save cross-validation results
        cv_results_dir = f"{DIR_Data}/cross_validation_results/"
        os.makedirs(cv_results_dir, exist_ok=True)

        detailed_cv_results_file = os.path.join(
            cv_results_dir, f"detailed_cv_results_{len(num_evaluation_seeds)}_seeds.xlsx")
        summary_cv_results_file = os.path.join(
            cv_results_dir, f"summary_cv_statistics_{len(num_evaluation_seeds)}_seeds.xlsx")

        df_all_cv_results.to_excel(detailed_cv_results_file, index=False)
        summary_stats.to_excel(summary_cv_results_file, index=False)

        with open(evaluation_log_file, "a") as log:
            log.write(
                f"\nCross-validation completed: {datetime.datetime.now()}\n")
            log.write(
                f"Updated Phase 1 Excel files with Mean_CV_R2_Test column\n")
            log.write(
                f"Detailed CV results saved to: {detailed_cv_results_file}\n")
            log.write(
                f"Summary CV statistics saved to: {summary_cv_results_file}\n")
            log.write("="*50 + "\n")

        print(f"Phase 2 completed!")
        print(f"Phase 1 Excel files updated with Mean_CV_R2_Test column")
        print(f"Detailed CV results: {detailed_cv_results_file}")
        print(f"Summary CV statistics: {summary_cv_results_file}")

        # CHANGED: Store lightweight results instead of heavy model objects
        best_models['cv_results'] = df_all_cv_results
        best_models['cv_summary_statistics'] = summary_stats
        best_models['mean_cv_scores'] = mean_cv_scores
    else:
        print("No CV results generated")
        best_models['cv_results'] = None
        best_models['cv_summary_statistics'] = None
        best_models['mean_cv_scores'] = None

    # CHANGED: Store only file paths and metadata, not actual models
    best_models['trained_models_info'] = saved_models_info

    print("=" * 60)
    print("FUNCTION COMPLETION")
    print("=" * 60)
    print(f"best_models keys: {list(best_models.keys())}")
    print("Function about to return...")

    return best_models


def Training_Testing_models(listofmodels, X, Y, groups, X_test, Y_test, groups_test, foldName, log_file=None):
    """Train and evaluate ML models for Specific Activity prediction.

    This function implements the core training and testing pipeline for SA prediction,
    training multiple ML algorithms on spectroscopic features and evaluating their
    performance on held-out test data.

    Args:
        listofmodels (list): List of sklearn-compatible ML models to train
        X (array-like): Training feature matrix (concatenated spectroscopic data)
        Y (array-like): Training target values (Specific Activity measurements in A/m²-Pt)
        groups (array-like): Group labels for samples (used for grouped splitting)
        X_test (array-like): Test feature matrix
        Y_test (array-like): Test target values
        groups_test (array-like): Test group labels
        foldName (str): Identifier for the current fold/experiment
        log_file (str, optional): Path to log file for training progress

    Returns:
        tuple: (results_dataframe, trained_models_dict)
            - results_dataframe: Performance metrics for all models
            - trained_models_dict: Dictionary of trained model objects

    Key Metrics Calculated:
        - R² score on training and test sets
        - Mean Absolute Error (MAE)
        - Feature importance rankings
        - Model-specific parameters and metadata
    """
    X_train = np.array(X)
    Y_train = np.array(Y)
    groups = np.array(groups)

    print(f"Training data shape: {X.shape}")

    output_filename = f"{DIR_Data}/models_scores_SA/"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    results = []
    # REMOVED: models_dict = {}  # This was consuming massive memory

    print(
        f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Log model training start
    if log_file:
        with open(log_file, "a") as log:
            log.write(f"  Model Training for {foldName}\n")
            log.write(
                f"  Training shape: {X_train.shape}, Test shape: {np.array(X_test).shape}\n")

    for index, original_model in enumerate(listofmodels):
        model = clone(original_model)
        print("fitting model")
        model.fit(X_train, Y_train)
        print("prediction")
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)
        print("metrics")
        # Metrics
        train_score = r2_score(Y_train, Y_pred_train)
        test_score = r2_score(Y_test, Y_pred_test)
        mae_train = mean_absolute_error(Y_train, Y_pred_train)
        mae_test = mean_absolute_error(Y_test, Y_pred_test)


        model_result = {
            'Model': model.__class__.__name__,
            'R2 Train': train_score,
            'R2 Test': test_score,
            'MAE Train': mae_train,
            'MAE Test': mae_test,
        }

        results.append(model_result)

        # Log model metrics
        if log_file:
            with open(log_file, "a") as log:
                log.write(f"    {model.__class__.__name__}:\n")
                log.write(
                    f"      R2 Train: {train_score:.4f}, R2 Test: {test_score:.4f}\n")
                log.write(
                    f"      MAE Train: {mae_train:.4f}, MAE Test: {mae_test:.4f}\n")

        # Save model to file
        model_filename = f"{output_filename}{foldName}_model_{index+1}_{model.__class__.__name__}_SA.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

    print("savingresults")
    df_results = pd.DataFrame(results)
    df_results.to_excel(
        f"{output_filename}{foldName}_SA_model_mse_scores.xlsx", index=False)
    print("Models saved and evaluation results written to Excel.")

    return {}  # Models are saved to files, no need to return them in memory

# Function to resample data on equivalent spce grid
def resampleDataset(DataDictionary, DataNameLitral, DataArray, chartechnique, NormFlag):
    # local Dataset according to the min/max/fixed data length
    localDataArray = {}
    # Function to get the data for respective Datatype -- [Functiona Signature] four parameter needed
    minElement, maxElement, localDataArray, key_max, key_min = getDefaultData(
        localDataArray, DataDictionary, DataNameLitral, True, DataArray[chartechnique])
    # Function to interpolateddata/extrapoloteddata [Functiona Signature] two parameter needed
    data_New, data_New_XY = getInterXtrapolatedata(
        localDataArray, InterXtrapolateFlag, NormFlag, minElement, maxElement, key_max, key_min)
    return data_New, data_New_XY

# Function to resample data on equivalent spce grid
def resampleDatasetfixed(dictkey, DataArray_New):
    Data_NewF = {}
    Data_New_XYF = {}
    desiredlength = 300
    NormFlag = True
    for keys, data in enumerate(DataArray_New[dictkey]):
        if len(DataArray_New[dictkey][data][0]) == 0:
            Data_NewF[data] = DataArray_New[dictkey][data][0]
            Data_New_XYF[data] = [DataArray_New[dictkey]
                                  [data][0], DataArray_New[dictkey][data][0]]
        else:
            regrid = np.linspace(
                0, DataArray_New[dictkey][data][0].size, desiredlength).astype(float)
            resample, new_x = fixedlendata(
                DataArray_New[dictkey][data][0], DataArray_New[dictkey][data][1], regrid, type=1)
            if NormFlag == False:
                Data_NewF[data] = resample
                Data_New_XYF[data] = [new_x, resample]
            if NormFlag == True:
                # Data_New[keys] = normalize_data(data =  resample.tolist())
                Data_NewF[data] = normalize_data(resample)
                Data_New_XYF[data] = [new_x, normalize_data(resample)]
    return Data_NewF, Data_New_XYF


def get_number_from_prefix(filename):
    # Define the mapping of prefixes to numbers
    prefix_map = {
        "Double": 2,
        "Triple": 3,
        "Quad": 4,
        "Penta": 5,
        "Hexa": 6,
        "Hepta": 7,
        "Octa": 8
    }

    # Iterate through the prefix map and check if the filename starts with any of the keys
    for prefix, number in prefix_map.items():
        if filename.startswith(prefix):
            return number
    return None

# Fuction to Creat the Possible Permutations of the dataset
def data_get(models, DataArray, DataArray_New, x_lst, y_lst, test_x_lst, test_y_lst, customsequence='', sampledetails={}):
    dataList = []
    for key in DataArray:
        dataList.append(key)

    dataList_New = []
    for key in DataArray_New:
        dataList_New.append(key)

    if customsequence != '':
        sequence = customsequence
        string_key = ''
        for element in sequence:
            string_key += str(element)
        string_key = string_key.replace('(', '').replace(')', '').replace(
            ',', '_').replace(' ', '').replace('_New', '_')
        if string_key.endswith('_'):
            string_key = string_key[:-1]
        x_train_temp = []
        x_test_temp = []
        # for index in range(0,1):
        for index in range(len(sequence)):
            X_temp_train, y_temp_train = prepareXandYdata(
                x_lst, DataArray_New[sequence[index]], y_lst)
            X_temp_test, y_temp_test = prepareXandYdata(
                test_x_lst, DataArray_New[sequence[index]], test_y_lst)
            x_train_temp.append(X_temp_train)
            x_test_temp.append(X_temp_test)

        # Concatenate X and Y train/test data
        X_train = customConcatinate(x_train_temp)
        X_test = customConcatinate(x_test_temp)

        # Assuming allModel is a result generated by the function
        return models, X_train, y_lst, X_test, test_y_lst


def query_feature_importance(feature_name, df):
    """
    Queries the feature importance values for a given feature from the results DataFrame,
    showing the full Custom Sequence, and the specific mean and std for the queried feature.

    Parameters:
        feature_name (str): The feature to search for (e.g., 'EXAFS').
        df (pd.DataFrame): DataFrame containing feature importance results.

    Returns:
        pd.DataFrame: DataFrame with the full sequence, mean arrays, and highlighted
                    mean/std for the queried feature, along with overall mean and std.
    """
    # Filter rows where the 'Custom Sequence' contains the feature
    feature_results = df[df['Custom Sequence'].apply(
        lambda x: feature_name in x)]

    # Initialize lists to store the extracted feature-specific values
    feature_means = []

    # Iterate over each row in the filtered results
    for _, row in feature_results.iterrows():
        custom_sequence = row['Custom Sequence']

        # Find the index of the feature in the Custom Sequence
        feature_index = custom_sequence.index(feature_name)

        # Retrieve the specific mean for the feature
        feature_means.append(row['feature_contribution'][feature_index])

    # Calculate the overall mean and standard deviation for the feature's contributions
    feature_mean_value = np.mean(feature_means)
    feature_std_value = np.std(feature_means)

    # Construct the final DataFrame with detailed output
    final_results = pd.DataFrame({
        'Custom Sequence': feature_results['Custom Sequence'],
        f'{feature_name} Mean': feature_means,
    })

    # Append the overall mean and standard deviation as the last row
    summary_row = pd.DataFrame({
        'Custom Sequence': ['Overall Mean/Std'],
        f'{feature_name} Mean': [f'Mean: {feature_mean_value:.5f}, Std: {feature_std_value:.5f}']
    })

    # Concatenate the results with the summary row
    final_results = pd.concat([final_results, summary_row], ignore_index=True)

    return final_results


# Code to estimate the posterior mean
def gp_posterior_mean(K, y, sigma):
    identity_matrix = np.identity(K.shape[0])
    return np.matmul(np.matmul(K, np.linalg.inv(K + sigma**2 * identity_matrix)), y)

# This code is to mix the two spectrum and generage new spectrum based on the
def spectral_mixup(sp, n_spec=2, alpha=0.5, quantity=1):
    """
    Randomly generates new spectra by mixing together several spectra with
    a Dirichlet probability distribution.

    This function is inspired of the Mixeup method proposed by zang (Zhang, Hongyi, et al. 2017).

    Notes:
        Updated [2023-05-31]:
            - parameter `mode` removed, use `return_infos` instead for parameters selection and validation.
            - Computation time and memory consumption reduced !

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels(must be binary) assigned the "sp" spectra, array shape = (n_spectra, n_classes).

        n_spec: integer, default=2
            Amount of spectrum mixed together.

        alpha : float
            Dirichlet distribution concentration parameter.

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

        return_infos : boolean, default=False
            If True, returns the indexes and the lambda values of the spectra mixed together

    Return:
        (array) New spectra generated.

        (array) New labels generated.

        (array) Optional; Indexes of the spectra mixed together.

        (array) Optional; Lambda values of the spectra mixed together.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length
    # array preallocation
    sp_aug = np.zeros((quantity * n_spectra, sp_len))
    # initialization and space allocation
    alpha_array = np.ones(n_spec) * alpha
    # Lambda values generated with a dirichlet distribution
    lambda_values = np.random.dirichlet(alpha_array, quantity*n_spectra)
    # random spectra index selection
    random_indexes = np.random.choice(n_spectra, size=(
        quantity * n_spectra, n_spec), replace=True)

    for i, (lam, index) in enumerate(zip(lambda_values, random_indexes)):
        mixed_sp = lam[:, np.newaxis] * sp[index]
        sp_aug[i] += np.sum(mixed_sp, axis=0)
    return sp_aug

# Code to estimate the posterior con matrix
def gp_posterior_cov(K, sigma):
    identity_matrix = np.identity(K.shape[0])
    return sigma**2 * np.matmul(K, np.linalg.inv(K + sigma**2 * identity_matrix))


def Data_FI(DataArray, DataArray_New, x_lst, y_lst, customsequence = ''):
    dataList = []
    for key in DataArray:
        dataList.append(key)

    dataList_New = []
    for key in DataArray_New:
        dataList_New.append(key)

    if customsequence != '':
        sequence = customsequence 
        # print(sequence)
        if len(sequence) == 1:
            foldName = 'SingleDataSet'
        if len(sequence) == 2:
            foldName = 'DoubleDataSet'
        if len(sequence) == 3:
            foldName = 'TripleDataSet'
        if len(sequence) == 4:
            foldName = 'QuadDataSet'
        if len(sequence) == 5:
            foldName = 'PentaDataSet'
        if len(sequence) == 6:
            foldName = 'HexaDataSet'
        if len(sequence) == 7:
            foldName = 'HeptaDataSet'
        if len(sequence) == 8:
            foldName = 'OctaDataSet'
        # concatenate elements into a single string
        string_key = ''
        for element in sequence:
            string_key += str(element)
        string_key = string_key.replace('(', '').replace(')', '').replace(',', '_').replace(' ', '').replace('_New', '_')
        if string_key.endswith('_'):
            string_key = string_key[:-1]
        # print(string_key)
        x_train_temp = []
        # for index in range(0,1):
        for index in range(len(sequence)):
            X_temp_train, y_temp_train = prepareXandYdata(x_lst, DataArray_New[sequence[index]], y_lst)
            x_train_temp.append(X_temp_train)

        # Concatenate X and Y train data
        X_train = customConcatinate(x_train_temp)

        # Assuming allModel is a result generated by the function
        return X_train, y_lst


# Program starts from here..!
random.seed(42)
# Set up the file paths and list down the folder and dataset
folder_path = 'MM_dataset_II_III_IV_V_Aug2024/'
filetype = '.xlsx'
CharArray = ['XANES', 'EXAFS_K2', 'XRD_2_P', 'PDF',
             'SAXS', 'HAXPES_VB', 'HAXPES_Pt4f', 'HAXPES_Pt3d']
DataArray = {'XANES': 0, 'EXAFS_K2': 1, 'XRD_2_P': 2, 'PDF': 3,
             'SAXS': 4, 'HAXPES_VB': 5, 'HAXPES_Pt4f': 6, 'HAXPES_Pt3d': 7}
FileLitral = ['G2MIRAI_E', 'G2MIRAI_V', 'TEC10E30E', 'TEC10EA50E', 'TEC10F30E', 'TEC10F50E', 'TEC10F50E_HT', 'TEC10V30E', 'TEC10V50E', 'TEC35V31E', 'TEC36E52', 'TEC36F52',
              'IP01_210903_1', 'IP01_220920_1', 'IP02_210204_2', 'IP02_211206_1', 'IP17_211213_1', 'PE01_210728_1', 'PE01_210728_2', 'PE01_210728_3', 'PE01_230925_4', 'PE01_230925_5', 'PE01_230925_6', 'PE02_220214_1', 'PE02_230424_1',
              'FCPF_EC_240111_1', 'FCPF_EC_240111_2', 'FCPF_EC_240111_3', 'FCPF_EC_240111_4', 'FCPF_EC_240111_5', 'FCPF_EC_240111_6', 'FCPF_EC_240111_7', 'FCPF_EC_240111_8', 'FCPF_EC_240111_9',
              'FCPF_MA_240522_1_H', 'FCPF_MA_240522_2_H', 'FCPF_MA_240522_3_H', 'FCPF_MA_240522_4_H', 'FCPF_MA_240522_6_H', 'FCPF_MA_240522_7_H', 'FCPF_MA_240522_8_H',  # 'FCPF_MA_240522_5_H.xlsx',
              'FCPF_MA_240522_9_H', 'FCPF_MA_240522_10_H', 'FCPF_MA_240522_11_H', 'FCPF_MA_240522_12_H', 'FCPF_MA_240522_15_H', ]

dataKeys = ['Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Data_8', 'Data_9', 'Data_10', 'Data_11', 'Data_12',
            'Data_13', 'Data_14', 'Data_15', 'Data_16', 'Data_17', 'Data_18', 'Data_19', 'Data_20', 'Data_21', 'Data_22', 'Data_23', 'Data_24', 'Data_25',
            'Data_26', 'Data_27', 'Data_28', 'Data_29', 'Data_30', 'Data_31', 'Data_32', 'Data_33', 'Data_34',
            'Data_35', 'Data_36', 'Data_37', 'Data_38', 'Data_40', 'Data_41', 'Data_42',  # 'Data_39',
            'Data_43', 'Data_44', 'Data_45', 'Data_46', 'Data_47']

Data_dict = dict.fromkeys(FileLitral)

# list of ML model that would like to test
models = [RandomForestRegressor(), MLPRegressor(max_iter=5000), DecisionTreeRegressor(), XGBRegressor(), Ridge(alpha=10.0)]
MLmodellist = listofMLmodels(models)

# Creating paths for all kind of files
PathDict = {}
for index, data in enumerate(dataKeys):
    Pathlist = []
    for pathname in DataArray.keys():
        Pathlist.append(folder_path+pathname+'/'+FileLitral[index]+filetype)
    Litral = FileLitral[index]
    PathDict[Litral] = Pathlist

# Read Data for all kind of paths
DataDictionary = {}
for Keys in PathDict.keys():
    readData = []
    for paths in PathDict[Keys]:
        print(paths)
        readData.append(ReadCsvData(paths))
    DataDictionary[Keys] = readData

# Following is the code to disable/enable the row normalization of RawData dictionary
# RawData Normalization is disable
normalizeFlag = False
if normalizeFlag == True:
    for dictindex, dictkey in enumerate(DataDictionary):
        for dataindex, datakey in enumerate(DataArray):
            if any(DataDictionary[dictkey][DataArray[datakey]]):
                DataDictionary[dictkey][DataArray[datakey]][1] = normalize_data(
                    DataDictionary[dictkey][DataArray[datakey]][1])

# # Code to plot all the data with respect to the measurement
# plot is disable
plotDataFlag = False
if plotDataFlag == True:
    for dictindex, dictkey in enumerate(DataDictionary):
        for dataindex, datakey in enumerate(DataArray):
            # Prepare the figure to plot the any kind of data
            if any(DataDictionary[dictkey][DataArray[datakey]]):
                PlotCustumFigure(DataDictionary[dictkey][DataArray[datakey]][1], dictkey, datakey,
                                 XaxisFlag=True, xdata=DataDictionary[dictkey][DataArray[datakey]][0], RangeStart=0, RangeEnd=np.array(DataDictionary[dictkey][DataArray[datakey]][1]).shape[0],
                                 NormFlag=False)
            else:
                PlotCustumFigure([], dictkey, datakey,
                                 XaxisFlag=False, RangeStart=0, RangeEnd=0,
                                 NormFlag=False)

DataNameLitral = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47]  # 39,
# Following options are available for interpolation of the RawData
InterXtrapolateFlag = 'Inter'   # |'fixed' 'Xtra' 'Inter'

# Normalization Flag is enable
NormFlag = False
# Process the each dataset one by one to resample on new x-axis grid
DataArray_New = {}
DataArray_New_XY = {}
for chaterization in CharArray:
    Data_New, Data_New_XY = resampleDataset(
        DataDictionary, DataNameLitral, DataArray, chaterization, NormFlag)
    DataArray_New[chaterization] = Data_New
    DataArray_New_XY[chaterization] = Data_New_XY


# Normalization Flag is enable
resamplingfixedCheck = True
# Process the each dataset one by one to resample on new x-axis grid [fixed]
if resamplingfixedCheck == True:
    DataArray_NewF = {}
    DataArray_New_XYF = {}
    for dictindex, dictkey in enumerate(DataArray_New_XY):
        Data_NewF, Data_New_XYF = resampleDatasetfixed(
            dictkey, DataArray_New_XY)
        DataArray_NewF[dictkey] = Data_NewF
        DataArray_New_XYF[dictkey] = Data_New_XYF
    DataArray_New = DataArray_NewF
    DataArray_New_XY = DataArray_New_XYF

customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF',
                  'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')

for i in customsequence:
    data_x = DataArray_New_XY[i]['y1'][0]
    data_x = data_x.T
    directory = f'{DIR_Data}/results_aug2024_I_II_III_IV_V/Analysis/'
    filename = f'{i}_Data_x.csv'
    filepath = os.path.join(directory, filename)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Write the data to the CSV file
    with open(filepath, 'w', newline='') as file:
        # data_x1 = data_x.reshape(-1,1)
        writer = csv.writer(file)
        writer.writerow(data_x)

# # Code to plot all the data with respect to the measurement with resalling plot
# plot is disable
plotDataFlag = True
if plotDataFlag == True:
    for dictindex, dictkey in enumerate(DataArray_New_XY):
        # Prepare the figure to plot the any kind of data
        if DataArray_New_XY[dictkey] != {}:
            PlotCustumFigureChar(DataArray_New_XY, dictkey)
        else:
            print('DataArray_New_XY is blank')


# Prepare the Training Dataset [Features]
sampleNotation = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12',
                  'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22', 'y23', 'y24', 'y25',
                  'y26', 'y27', 'y28', 'y29', 'y30', 'y31', 'y32', 'y33', 'y34',
                  'y35', 'y36', 'y37', 'y38',  'y40', 'y41', 'y42',  # 'y39',
                  'y43', 'y44', 'y45', 'y46', 'y47']
sampleNotation_dict = dict(zip(sampleNotation, FileLitral))

train_x_lst = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12',
               'y13', 'y14', 'y15', 'y16', 'y17', 'y18',  'y19',  'y20',  'y21', 'y22', 'y23', 'y24', 'y25',
               'y26', 'y27', 'y28', 'y29', 'y30', 'y31', 'y32', 'y33', 'y34',
               'y35', 'y36', 'y37', 'y38',  'y40', 'y41', 'y42',  # 'y39',
               'y43', 'y44', 'y45', 'y46', 'y47']

train_y_lst = [10.9, 8.5, 5.7, 6, 4.4, 5.2, 3.6, 4.9, 6, 8, 17.1, 16.3,
                8.1, 7.8, 28, 32, 5.5, 7, 9.1, 9.7, 7.2, 8.1, 8.2, 7.2, 15.6,
                3.8, 4.1, 4.6, 3.6, 3.2, 3.5, 12.8, 12, 13.4,
                8.76, 7.35, 8.08, 7.18, 7.01, 23.64, 19.5,
                22.72, 19.79, 12.28, 7.67, 5.74]

def get_filename(index):
    filenames = ['SingleDataSet', 'DoubleDataSet', 'TripleDataSet', 'QuadDataSet',
                 'PentaDataSet', 'HexaDataSet', 'HeptaDataSet', 'OctaDataSet']
    return filenames[index]


# Custom sequence
customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF',
                  'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')

customflag = False
if customflag == True:
    try:
        print("Starting checkPermutationsnew...")
        models_dict = checkPermutationsnew(
            MLmodellist, DataArray_New, train_x_lst, train_y_lst)
        print("checkPermutationsnew completed successfully!")
    except:
        print("error while training")

Ridge_model = r"models_scores_SA/OctaDataSet_Seed334_model_1_Ridge_SA.pkl"
XGB_model = r"models_scores_SA/OctaDataSet_Seed334_model_4_XGBRegressor_SA.pkl"

models_list = [ f"{DIR_Data}/{Ridge_model}", f"{DIR_Data}/{XGB_model}",]

# Iterate over each permutation
for i in models_list:
    selected_data = customsequence
    with open(i, 'rb') as file:
        loaded_model = pickle.load(file)
    # Get filename based on custom sequence length (or another criteria)
    fname = get_filename(len(selected_data) - 1)

    # Check permutations with the selected datasets (assuming checkPermutations is already defined)
    X_train, Y_train = Data_FI(
        DataArray, DataArray_New, train_x_lst, train_y_lst, 
        customsequence=selected_data
    )
    # Get permutation importance of features
    result = permutation_importance(loaded_model, X_train, Y_train, scoring='neg_mean_squared_error', n_repeats=5, n_jobs=15)
    importances_mean = np.round(result.importances_mean, 5)
    importances_mean=importances_mean.T

    importances_mean = pd.DataFrame(importances_mean)
    importances_mean.to_excel(f"{DIR_Data}/{str(type(loaded_model).__name__)}_Feature_importance_SA.xlsx")

    results_data = []
    
    prefix = get_number_from_prefix(i)
    # Generate all permutations of customsequence
    permutations = list(itertools.permutations(customsequence, prefix))
    x = 0
    # Iterate over each permutation
    for selected_data in permutations:
        # Get filename based on custom sequence length (or another criteria)
        fname = get_filename(len(selected_data) - 1)
        x = x + 1
        print(x)
        # Check permutations with the selected datasets (assuming checkPermutations is already defined)
        X_train, Y_train = Data_FI(
        DataArray, DataArray_New, train_x_lst, train_y_lst, 
        customsequence=selected_data
    )

        # Get permutation importance of features
        result = permutation_importance(loaded_model, X_train, Y_train, scoring='neg_mean_squared_error', n_repeats=5, n_jobs=50)
        importances_mean = np.round(result.importances_mean, 5)
        importances_std = result.importances_std

        # Calculate feature contribution (sum of each 300-length chunk of importances)
        feature_contribution = [sum(importances_mean[i:i + 300]) for i in range(0, len(importances_mean), 300)]
        feature_contribution_total = sum(feature_contribution)

        # Calculate feature percentages
        feature_percentages = [(contribution / feature_contribution_total) * 100 for contribution in feature_contribution]

        # Collect the results to be saved to Excel
        results_data.append({
                'Custom Sequence': selected_data,
                'feature_contribution': feature_percentages,
                # 'feature_contribution sum': feature_contribution_total

            })

    # Convert results to a pandas DataFrame
    results_df = pd.DataFrame(results_data)

    # Save the DataFrame to an Excel file
    output_filename = f"importance_results_{i}.xlsx"
    results_df = results_df.apply(pd.to_numeric, errors='ignore')
    results_df.to_excel(output_filename, index=False)

    print(f"Feature importances saved to {output_filename}")


    for j in customsequence:

    # Example usage:
        feature_name = j

        feature_importance_results = query_feature_importance(feature_name, results_df)

        # Print the results for the feature
        print(feature_importance_results)

        # Optionally, save the results to a separate Excel file
        feature_importance_results.to_excel(f"{i}_{feature_name}_importance_query_results.xlsx", index=False)

# # This code is to mix the two spectra and test on newly traine model.
spectra_XAFS = X_train[:,0:300]
spectra_XRD = X_train[:,300:600]

# This code is to generate the XAFS using the Gaussian Process
desiredspectrum = 1000//len(spectra_XAFS)
x = np.linspace(0, spectra_XAFS.shape[1], num=spectra_XAFS.shape[1]).reshape(-1, 1)
sigma = 0.05 #0.01
K_rbf = rbf_kernel(x, gamma = .01)


spectral_mixup_XAFS = np.empty((0, 0))
for index, data in enumerate(spectra_XAFS):
    gp_mean = gp_posterior_mean(K_rbf, data.reshape(-1, 1), sigma)[:,0]
    gp_cov = gp_posterior_cov(K_rbf, sigma)
    gp_sample = np.random.multivariate_normal(gp_mean, gp_cov, desiredspectrum)
    # Check if the ndarray is empty
    if spectral_mixup_XAFS.size == 0:
        spectral_mixup_XAFS = gp_sample
    else:
        print("The ndarray is not empty")
        spectral_mixup_XAFS = np.vstack((spectral_mixup_XAFS, gp_sample))

# This code is to generate the XRD using the Gaussian Process
#Design the Kernel same size as data with differnt 'gamma' and try
x = np.linspace(0, spectra_XAFS.shape[1], num=spectra_XAFS.shape[1]).reshape(-1, 1)
sigma = 0.01 #0.08
K_rbf = rbf_kernel(x, gamma = .08)

spectral_mixup_XRD = np.empty((0, 0))
for index, data in enumerate(spectra_XRD):
    gp_mean = gp_posterior_mean(K_rbf, data.reshape(-1, 1), sigma)[:,0]
    gp_cov = gp_posterior_cov(K_rbf, sigma)
    gp_sample = np.random.multivariate_normal(gp_mean, gp_cov, desiredspectrum)
    # Check if the ndarray is empty
    if spectral_mixup_XRD.size == 0:
        spectral_mixup_XRD = gp_sample
    else:
        print("The ndarray is not empty")
        spectral_mixup_XRD = np.vstack((spectral_mixup_XRD, gp_sample))

spectral_mixup_XAFS = spectral_mixup(spectral_mixup_XAFS, n_spec=5, alpha=0.01, quantity=1)
spectral_mixup_XRD = spectral_mixup(spectral_mixup_XRD, n_spec=5, alpha=0.3, quantity=1)


New_Xtest = np.zeros((len(spectral_mixup_XAFS), 2400))
# This code appends the artificial data to the conventional data
counter = 0

log_file = f"{DIR_Data}/Augmented_data.txt"
 
with open(log_file, 'a') as file:
    file.write(f'Index, Predicted score, Augmented data \n')
    # Prepare the augmented data
    augmented_data_list = []
    for index, (xafs, xrd) in enumerate(zip(spectral_mixup_XAFS, spectral_mixup_XRD)):
        if New_Xtest.shape[0] == index:
            break
        else:
            augmented_data = np.concatenate((xafs, xrd, X_train[counter, 600:]), axis=0)
            New_Xtest[index] = np.round(augmented_data,5)
           
            augmented_data_list.append(augmented_data)
           
            counter = counter + 1
            if counter == len(dataKeys):
                counter = 0
 
    # Predict scores and log data
    models_list = [ f"{DIR_Data}/{Ridge_model}"]

    # Iterate over each permutation
    for i in models_list:
        selected_data = customsequence
        with open(i, 'rb') as file:
            loaded_model = pickle.load(file)
 
            # Predict scores
            predictedscore1 = loaded_model.predict(New_Xtest)
           
            # Log the predicted data and augmented data
            file.write(f"Model: {type(loaded_model).__name__}\n")
            for index, (testdata, aug_data) in enumerate(zip(predictedscore1, augmented_data_list)):
                file.write(f'{index},{testdata},{",".join(map(str, aug_data))}\n')