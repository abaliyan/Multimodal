import os
import itertools
import numpy as np
import pandas as pd

import pickle
import random
import csv

import matplotlib
from sklearn.metrics import r2_score
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import Counter

from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from sklearn.metrics.pairwise import rbf_kernel

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from itertools import permutations
from scipy.interpolate import interp1d
from spectres import spectres
import scipy.interpolate

import datetime
import pickle

from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, train_test_split

DIR_Data = "models_test"

# Function to find the mising files
def find_missing_files(list1, list2):
    return list(set(list1) - set(list2))

# Function to get the time stamp
def getTime ():
    # return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return datetime.date.today().strftime("%Y-%m-%d")

# Function to normalize the column vector
def normalize_data(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

# Function to read the csv file
def ReadCsvData(filename):
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
            Localfiledata = [df[0].tolist()[0:281], df[1].tolist()[0:281]] #Localfiledata = [df[0].tolist()[0:281], df[1].tolist()[0:281]]
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
def PlotCustumFigure(data, foldername = '', Data_type = '', XaxisFlag = True, xdata = np.empty(0), RangeStart = 0, RangeEnd = 0, NormFlag = True):
    if len(data) != 0 and np.isnan(data).any() == False:
        fig = plt.figure(1, figsize=(5, 4))
        # Code to shorten the data range [Min, max]
        data = data[RangeStart:RangeEnd]

        if  NormFlag == True:
            data= normalize_data(data)
        if  XaxisFlag == True:
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
                custumtitle =  Data_type
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
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=20)
        plt.axis('off')
        if Data_type != '':
        # custumtitle = foldername+  '_'+ Data_type
            custumtitle =  Data_type
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
            if  NormFlag == True:
                ydata= normalize_data(ydata)
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
def PlotCustumFigureMulti(data, foldername = '', Data_type = '', XaxisFlag = True, RangeStart = 0, RangeEnd = 0, NormFlag = True):
    if len(data) != 0:
        fig = plt.figure(1, figsize=(5, 4))
        # Code to shorten the data range [Min, max]
        data = data[RangeStart:RangeEnd]

        if  NormFlag == True:
            data= normalize_data(data)
        if  XaxisFlag == True:
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
                custumtitle =  Data_type
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
def save_figure_to_folder(figure, folder_name='', file_name = ''):
    # define the name of the directory to be created
    if folder_name != '':
        localresultpath = f"{DIR_Data}" + '/'+ folder_name
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
def save_figure_to_folderMulti(figure, folder_name='', file_name = ''):
    # define the name of the directory to be created
    if folder_name != '':
        localresultpath = f"{DIR_Data}" + '/'+ folder_name
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
                DatatypeName['y' + str(DataNameLitral[counter])] = convert_to_array(DataDictionary[datakeys][datatype][1]).flatten()
        else:
            DatatypeName['y' + str(DataNameLitral[counter])] = convert_to_array(DataDictionary[datakeys][datatype][0])
        counter = counter+1

    # check the minimum and maximum of the list of arrays for interpolate/extrapolate
    rows = [len(row) for keys, row in DatatypeName.items() if len(row) !=0]
    mykeys = [keys for keys, row in DatatypeName.items() if len(row) !=0]
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
            DatatypeName['y' + str(DataNameLitral[counter])] = convert_to_array(DataDictionary[datakeys][datatype][0])
        counter = counter+1

    return minElement, maxElement, DatatypeName, mykeys_max, mykeys_min

# Function to Interpolate Data
def interpolatedata (old_x, old_y, new_x, type = 'linear'):
    f = scipy.interpolate.interp1d(old_x, old_y, kind = type, fill_value='extrapolate')
    result = f(new_x)
    return result, new_x

# Function to extrapolate Data
def extrapolatedata (old_x, old_y, new_x, type = 1 ):
    f = interp1d(old_x, old_y, type, fill_value='extrapolate')
    new_x = np.linspace(new_x[0], new_x[-1], len(new_x))
    result = f(new_x)
    return result, new_x

# Function to extrapolate Data
def fixedlendata (old_x, old_y, new_x, type = 1 ):
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
                extrapoloteddata, new_x = extrapolatedata (data[0], data[1], Data_old[key_max][0], type = 1)
                if NormFlag == False:
                    Data_New[keys] = extrapoloteddata
                    Data_New_XY[keys]= [new_x, extrapoloteddata]
                if NormFlag == True:
                    # Data_New[keys] = normalize_data(data = extrapoloteddata.tolist())
                    Data_New[keys] = normalize_data(extrapoloteddata)
                    Data_New_XY[keys]= [new_x, normalize_data(extrapoloteddata)]
    if flag == 'Inter':
        for keys, data in Data_old.items():
            if len(data) == 0:
                Data_New[keys] = data
                Data_New_XY[keys] = [data, data] 
            else:
                print(keys)
                interpolateddata, new_x = interpolatedata (data[0], data[1], Data_old[key_min][0], type = 'linear') #interpolatedata(data[0], data[1], Data_old[key_min][0], type = 'linear')
                if NormFlag == False:
                    Data_New[keys] = interpolateddata
                    Data_New_XY[keys] = [new_x, interpolateddata]
                if NormFlag == True:
                    # Data_New[keys] = normalize_data(data = interpolateddata.tolist())
                    Data_New[keys] = normalize_data(interpolateddata)
                    Data_New_XY[keys] = [new_x, normalize_data(interpolateddata)]
    if flag == 'fixed':
        desiredlength = 200
        for keys, data in Data_old.items():
            if len(data) == 0:
                Data_New[keys] = data
                Data_New_XY[keys] = [data, data] 
            else:
                regrid = np.linspace(0, data[0].size, desiredlength).astype(float)
                resample, new_x = fixedlendata(data[0], data[1], regrid, type = 1)
                if NormFlag == False:
                    Data_New[keys] = resample
                    Data_New_XY[keys]= [new_x, resample] 
                if NormFlag == True:
                    # Data_New[keys] = normalize_data(data =  resample.tolist())
                    Data_New[keys] = normalize_data(resample)
                    Data_New_XY[keys] = [new_x, normalize_data(resample)] 
    return Data_New, Data_New_XY

# Fuction that list the traing ML models that one would like to test it
def Training_Testing_models (listofmodels, X, Y, foldName):
    Modellist = ['Linear Regression Model:', 'Decision Tree Model:', 'Random Forest Model:',# 'Random Forest Model:',
                 'Neural Network Model:', 'XGB regressor model:']
    
    X = np.array(X)
    Y = np.array(Y)
    # Train the models using the training sets
    allModel = []
    output_filename = f"{DIR_Data}/models_scores_MA/"

    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    mse_list_train = []
    mse_list_test = []
    results = []
    models_dict={}
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model_data_list = []
    for index, original_model in enumerate(listofmodels):
        # Create a fresh copy for this specific dataset
        model = clone(original_model)
        
        model.fit(X_train, Y_train)
        model_data_list.append({
            'model': model,  # This is now a unique model instance
            'X_train': X_train,
            'Y_train': Y_train
        })
        # allModel.append(model.fit(X_train, Y_train))
        Y_pred_test = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        train_score  = r2_score(Y_train, Y_pred_train)
        test_score  = r2_score(Y_test, Y_pred_test)
        # Calculate Mean Squared Error on the test set
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        mse_list_train.append(mse_train)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        mse_list_test.append(mse_test)
        shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        # Perform cross-validation on the whole dataset
        cv_scores = cross_val_score(model, X, Y, cv=shuffle_split, n_jobs=50)  # 5-fold cross-validation

        results.append({
            'Model': model.__class__.__name__,
            'mse_train': mse_train,
            'train_score': train_score,
            'mse_test': mse_test,
            'test_score':test_score,
            'Mean CV Score': cv_scores.mean(),
            'CV Std': cv_scores.std(),
        })

        model_filename = f"{output_filename}{foldName}_model_{index+1}_{model.__class__.__name__}_MA.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        models_dict[model_filename] = model
    df_results = pd.DataFrame(results)
    
    # Save the DataFrame to an Excel file
    df_results.to_excel(f"{output_filename}{foldName}_MA_model_mse_scores.xlsx", index=False)

    print("Models saved in .pkl format and results saved to model_mse_scores.xlsx")
   
    models_mse_dict = {"Linear Regression": mse_list_test[0],}
                    # "Decision Tree": mse_list_test[1],
                    # "Random Forest": mse_list_test[1],
                    # "Neural Network": mse_list_test[2],
                    # "XGB":mse_list_test[3]}
    
    return model_data_list

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
def plotTrainResult (train, test, gtest, legendtxt, foldername = '', Data_type = ''):
    
    if train.ndim == 1:
        fig =  plt.figure(figsize=(5, 5))
        if len(train) != 0:
            plt.plot(train,'ro',label="Train")
        if len(test) != 0:
            plt.plot(test,'bs',label='Test')
        if len(gtest) != 0:
            plt.plot(gtest,'gD',label='Gen1')
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
        ax1.plot([i[0] for i in train],'ro',label="Train")
        ax1.plot([i[0] for i in test],'bs',label='Test')
        ax1.set_title('ESCA')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot([i[1] for i in train],'ro',label="Train")
        ax2.plot([i[1] for i in test],'bs',label='Test')
        ax2.set_title('SA')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot([i[2] for i in train],'ro',label="Train")
        ax3.plot([i[2] for i in test],'bs',label='Test')
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

# Fuction to Creat the Possible Permutations of the dataset
def checkPermutationsnew(models, DataArray, DataArray_New, x_lst, y_lst, customsequence = '', sampledetails = {}):
    dataList = []
    for key in DataArray:
        dataList.append(key) #characterization name [no need]

    dataList_New = []
    for key in DataArray_New:
        dataList_New.append(key)

    if customsequence == '':
        totalDataXtrain = []
        totalDataYtrain = []
        bestModeldictName = {} 
        for pair in range(1,len(dataList_New)+1): 
            perm = list(permutations(dataList_New, pair))
            print(f'Total number of ways: {len(perm)}')
            for sequence in perm:
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
                string_key = ''
                for element in sequence:
                    string_key += str(element)
                string_key = string_key.replace('(', '').replace(')', '').replace(',', '_').replace(' ', '').replace('_New', '_')
                if string_key.endswith('_'):
                    string_key = string_key[:-1]
                x_train_temp = []
                y_test_temp = []
                for index in range(0,len(sequence)):
                    X_temp_train, y_temp_train = prepareXandYdata(x_lst, DataArray_New[sequence[index]], y_lst)
                    x_train_temp.append(X_temp_train)
                X_train = customConcatinate(x_train_temp)

                totalDataXtrain.extend(X_train)
                totalDataYtrain.extend(y_lst)
                savefig = False
                if savefig == True:
                    PlotCustumFigureMulti(X_train[0], foldername = foldName, Data_type = string_key,
                                    XaxisFlag = True, RangeStart = 0, RangeEnd = np.array(X_train[0]).shape[0], NormFlag = False)
            allModel = Training_Testing_models (models, totalDataXtrain, totalDataYtrain, foldName)
            bestModeldictName[foldName] = allModel
            totalDataXtrain = []
            totalDataYtrain = []
        return bestModeldictName
    

# Function to resample data on equivalent spce grid
def resampleDataset(DataDictionary, DataNameLitral, DataArray, chartechnique, NormFlag):
    # local Dataset according to the min/max/fixed data length 
    localDataArray = {}
    # Function to get the data for respective Datatype -- [Functiona Signature] four parameter needed
    minElement, maxElement, localDataArray, key_max, key_min = getDefaultData(localDataArray, DataDictionary, DataNameLitral, True, DataArray[chartechnique])
    # Function to interpolateddata/extrapoloteddata [Functiona Signature] two parameter needed
    data_New, data_New_XY = getInterXtrapolatedata(localDataArray,InterXtrapolateFlag, NormFlag, minElement, maxElement, key_max, key_min)
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
            Data_New_XYF[data] = [DataArray_New[dictkey][data][0],DataArray_New[dictkey][data][0]]
        else:
            regrid = np.linspace(0, DataArray_New[dictkey][data][0].size, desiredlength).astype(float)
            resample, new_x = fixedlendata(DataArray_New[dictkey][data][0], DataArray_New[dictkey][data][1], regrid, type = 1)
            if NormFlag == False:
                Data_NewF[data] = resample
                Data_New_XYF[data]= [new_x, resample] 
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
def data_get(models, DataArray, DataArray_New, x_lst, y_lst, test_x_lst, test_y_lst, customsequence = '', sampledetails = {}):
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
        string_key = string_key.replace('(', '').replace(')', '').replace(',', '_').replace(' ', '').replace('_New', '_')
        if string_key.endswith('_'):
            string_key = string_key[:-1]
        x_train_temp = []
        x_test_temp = []
        # for index in range(0,1):
        for index in range(len(sequence)):
            X_temp_train, y_temp_train = prepareXandYdata(x_lst, DataArray_New[sequence[index]], y_lst)
            X_temp_test, y_temp_test = prepareXandYdata(test_x_lst, DataArray_New[sequence[index]], test_y_lst)
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
    feature_results = df[df['Custom Sequence'].apply(lambda x: feature_name in x)]
    
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

    # Save the results to Excel
    output_filename = f"{feature_name}_importance_query_results.xlsx"
    # final_results.to_excel(output_filename, index=False)
    
    # print(f"Overall {feature_name} Mean: {feature_mean_value:.5f}")
    # print(f"Overall {feature_name} Std: {feature_std_value:.5f}")
    
    return final_results


# Code to estimate the posterior mean
def gp_posterior_mean(K, y, sigma):
    identity_matrix =  np.identity(K.shape[0])
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
    random_indexes = np.random.choice(n_spectra, size=(quantity * n_spectra, n_spec), replace=True)

    for i, (lam, index) in enumerate(zip(lambda_values, random_indexes)):
        mixed_sp = lam[:, np.newaxis] * sp[index]
        sp_aug[i] += np.sum(mixed_sp, axis=0)
    return sp_aug

# Code to estimate the posterior con matrix
def gp_posterior_cov(K, sigma):
    identity_matrix =  np.identity(K.shape[0])
    return sigma**2 * np.matmul(K, np.linalg.inv(K + sigma**2 * identity_matrix))

# Program starts from here..!
random.seed(42)
# Set up the file paths and list down the folder and dataset
folder_path = 'MM_dataset_II_III_IV_V_Aug2024/'
filetype = '.xlsx'
CharArray = ['XANES', 'EXAFS_K2','XRD_2_P','PDF','SAXS','HAXPES_VB','HAXPES_Pt4f', 'HAXPES_Pt3d']
DataArray = {'XANES':0, 'EXAFS_K2':1,'XRD_2_P':2,'PDF':3,'SAXS':4,'HAXPES_VB':5,'HAXPES_Pt4f':6, 'HAXPES_Pt3d':7}
FileLitral = ['G2MIRAI_E', 'G2MIRAI_V', 'TEC10E30E', 'TEC10EA50E', 'TEC10F30E','TEC10F50E','TEC10F50E_HT','TEC10V30E','TEC10V50E', 'TEC35V31E', 'TEC36E52', 'TEC36F52',
               'IP01_210903_1', 'IP01_220920_1', 'IP02_210204_2', 'IP02_211206_1', 'IP17_211213_1', 'PE01_210728_1', 'PE01_210728_2', 'PE01_210728_3', 'PE01_230925_4', 'PE01_230925_5', 'PE01_230925_6','PE02_220214_1', 'PE02_230424_1',
               'FCPF_EC_240111_1', 'FCPF_EC_240111_2', 'FCPF_EC_240111_3', 'FCPF_EC_240111_4', 'FCPF_EC_240111_5', 'FCPF_EC_240111_6', 'FCPF_EC_240111_7', 'FCPF_EC_240111_8', 'FCPF_EC_240111_9',
               'FCPF_MA_240522_1_H', 'FCPF_MA_240522_2_H', 'FCPF_MA_240522_3_H', 'FCPF_MA_240522_4_H', 'FCPF_MA_240522_6_H', 'FCPF_MA_240522_7_H', 'FCPF_MA_240522_8_H', # 'FCPF_MA_240522_5_H.xlsx',
               'FCPF_MA_240522_9_H', 'FCPF_MA_240522_10_H', 'FCPF_MA_240522_11_H', 'FCPF_MA_240522_12_H', 'FCPF_MA_240522_15_H', ]

dataKeys = ['Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Data_8', 'Data_9', 'Data_10', 'Data_11', 'Data_12',
            'Data_13', 'Data_14', 'Data_15', 'Data_16', 'Data_17', 'Data_18', 'Data_19', 'Data_20', 'Data_21','Data_22', 'Data_23', 'Data_24', 'Data_25',
            'Data_26', 'Data_27', 'Data_28', 'Data_29', 'Data_30', 'Data_31', 'Data_32', 'Data_33', 'Data_34',
            'Data_35', 'Data_36', 'Data_37', 'Data_38', 'Data_40', 'Data_41', 'Data_42', #'Data_39',
            'Data_43', 'Data_44', 'Data_45', 'Data_46', 'Data_47']

Data_dict = dict.fromkeys(FileLitral)

# list of ML model that would like to test
# models = [LinearRegression(),  DecisionTreeRegressor(), RandomForestRegressor(), MLPRegressor(max_iter=5000), XGBRegressor()]
models = [LinearRegression(),  DecisionTreeRegressor(), RandomForestRegressor(), MLPRegressor(max_iter=5000), XGBRegressor()]
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
                DataDictionary[dictkey][DataArray[datakey]][1] = normalize_data(DataDictionary[dictkey][DataArray[datakey]][1])

# # Code to plot all the data with respect to the measurement
# plot is disable 
plotDataFlag = False
if plotDataFlag == True:
    for dictindex, dictkey in enumerate(DataDictionary):
        for dataindex, datakey in enumerate(DataArray):
        # Prepare the figure to plot the any kind of data
            if any(DataDictionary[dictkey][DataArray[datakey]]):
                PlotCustumFigure(DataDictionary[dictkey][DataArray[datakey]][1], dictkey, datakey,
                                XaxisFlag = True, xdata = DataDictionary[dictkey][DataArray[datakey]][0], RangeStart = 0, RangeEnd = np.array(DataDictionary[dictkey][DataArray[datakey]][1]).shape[0], 
                                NormFlag = False)
            else:
                PlotCustumFigure([], dictkey, datakey,
                                XaxisFlag = False, RangeStart = 0, RangeEnd = 0, 
                                NormFlag = False)
                              
DataNameLitral = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47] # 39,
# Following options are available for interpolation of the RawData
InterXtrapolateFlag = 'Inter'   # |'fixed' 'Xtra' 'Inter'

# Normalization Flag is enable
NormFlag = False
# Process the each dataset one by one to resample on new x-axis grid
DataArray_New = {}
DataArray_New_XY = {}
for chaterization in CharArray:
    Data_New, Data_New_XY= resampleDataset(DataDictionary, DataNameLitral, DataArray, chaterization, NormFlag)
    DataArray_New[chaterization] = Data_New
    DataArray_New_XY[chaterization] = Data_New_XY


# Normalization Flag is enable
resamplingfixedCheck = True
# Process the each dataset one by one to resample on new x-axis grid [fixed]
if resamplingfixedCheck == True:
    DataArray_NewF = {}
    DataArray_New_XYF = {}
    for dictindex, dictkey in enumerate(DataArray_New_XY):
        Data_NewF, Data_New_XYF= resampleDatasetfixed(dictkey, DataArray_New_XY)
        DataArray_NewF[dictkey] = Data_NewF
        DataArray_New_XYF[dictkey] = Data_New_XYF
    DataArray_New = DataArray_NewF
    DataArray_New_XY = DataArray_New_XYF

customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF', 'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')

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
sampleNotation = ['y1', 'y2', 'y3', 'y4', 'y5','y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12',
                'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22', 'y23', 'y24', 'y25',
                'y26', 'y27','y28', 'y29', 'y30', 'y31', 'y32', 'y33', 'y34',
                'y35', 'y36', 'y37', 'y38',  'y40', 'y41', 'y42', #'y39',
                'y43', 'y44', 'y45', 'y46', 'y47']
sampleNotation_dict = dict(zip(sampleNotation, FileLitral))


train_x_lst = ['y1', 'y2', 'y3', 'y4', 'y5','y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12',
                'y13', 'y14', 'y15', 'y16', 'y17', 'y18',  'y19',  'y20',  'y21', 'y22', 'y23', 'y24', 'y25',
                'y26', 'y27','y28', 'y29', 'y30', 'y31', 'y32', 'y33', 'y34',
                'y35', 'y36', 'y37', 'y38',  'y40', 'y41', 'y42', #'y39',
                'y43', 'y44', 'y45', 'y46', 'y47']

train_y_lst = [490, 480, 540, 271, 360, 400, 175, 370, 340, 330, 530, 630,
                390, 380, 240, 315, 490, 811, 1066, 842, 670, 790, 842, 490, 610,
                324, 312, 319, 368, 323, 366, 385, 334, 365,
                841, 675, 776, 704, 701, 1300, 1053,
                1068, 772, 1990, 675, 436]

#for Feature Importance
test_x_lst = ['y8', 'y9', 'y11', 'y15', 'y19', 'y23', 'y25', 'y29', 'y34', 'y37', 'y43', 'y45', 'y47']
test_y_lst = [370, 340, 530, 240, 1066, 842, 610, 368, 365, 776, 1068, 1990, 436]

def get_filename(index):
    filenames = ['SingleDataSet', 'DoubleDataSet', 'TripleDataSet', 'QuadDataSet', 
                 'PentaDataSet', 'HexaDataSet', 'HeptaDataSet', 'OctaDataSet']
    return filenames[index]

# Custom sequence
customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF', 'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')

customflag = True
if customflag == True:
    models_dict = checkPermutationsnew(MLmodellist, DataArray, DataArray_New, train_x_lst, train_y_lst)
    for foldName, model_data_list in models_dict.items():
        if foldName == 'OctaDataSet':
            for model_data in model_data_list:
                model = model_data['model']
                X_train = model_data['X_train']
                Y_train = model_data['Y_train']
                results_data = []

                model_name = model.__class__.__name__
                prefix = get_number_from_prefix(foldName)  # if your foldName carries a number

                permutations = list(itertools.permutations(customsequence, prefix))

                for selected_data in permutations:
                    fname = get_filename(len(selected_data) - 1)

                    # Correctly pass single model
                    result = permutation_importance(
                        model, X_train, Y_train,
                        scoring='neg_mean_squared_error',
                        n_repeats=5, n_jobs=50
                    )

                    importances_mean = np.round(result.importances_mean, 5)
                    feature_contribution = [sum(importances_mean[i:i + 300]) for i in range(0, len(importances_mean), 300)]
                    total = sum(feature_contribution)
                    feature_percentages = [(x / total) * 100 if total != 0 else 0 for x in feature_contribution]
                    with open(f'{DIR_Data}/{fname}_feature_importance_MA.txt', 'a') as file:
                        file.write(f"Model: {type(model).__name__}\n")
                        file.write(','.join(map(str, importances_mean)) + '\n')

                    results_data.append({
                        'Custom Sequence': selected_data,
                        'feature_contribution': feature_percentages,
                    })

                # Save result
                results_df = pd.DataFrame(results_data)
                output_filename = f"{DIR_Data}/importance_results/importance_results_{prefix}_{foldName}_{model_name}.xlsx"
                output_dir = os.path.dirname(output_filename)
                os.makedirs(output_dir, exist_ok=True)
                results_df.to_excel(output_filename, index=False)

                print(f"Saved permutation importance to {output_filename}")
                        # Optional: Per-feature analysis
        

            for j in customsequence:

            # Example usage:
                feature_name = j

                feature_importance_results = query_feature_importance(feature_name, results_df)

                # Print the results for the feature
                print(feature_importance_results)

                output_filename = f"{DIR_Data}/importance_query_results/{model.__class__.__name__,}_{foldName}_{feature_name}_importance_query_results.xlsx"

                output_dir = os.path.dirname(output_filename)
                os.makedirs(output_dir, exist_ok=True)

                # Optionally, save the results to a separate Excel file
                feature_importance_results.to_excel(output_filename , index=False)
            # Perform augmentation
            spectra_XAFS = X_train[:, 0:300]
            spectra_XRD = X_train[:, 300:600]

            desiredspectrum = 1000 // len(spectra_XAFS)
            x = np.linspace(0, spectra_XAFS.shape[1], num=spectra_XAFS.shape[1]).reshape(-1, 1)

            # === AFS Augmentation ===
            K_rbf = rbf_kernel(x, gamma=.01)
            sigma = 0.05
            spectral_mixup_XAFS = np.vstack([
                np.random.multivariate_normal(
                    gp_posterior_mean(K_rbf, row.reshape(-1, 1), sigma)[:, 0],
                    gp_posterior_cov(K_rbf, sigma),
                    desiredspectrum
                ) for row in spectra_XAFS
            ])

            # === XRD Augmentation ===
            K_rbf = rbf_kernel(x, gamma=.08)
            sigma = 0.01
            spectral_mixup_XRD = np.vstack([
                np.random.multivariate_normal(
                    gp_posterior_mean(K_rbf, row.reshape(-1, 1), sigma)[:, 0],
                    gp_posterior_cov(K_rbf, sigma),
                    desiredspectrum
                ) for row in spectra_XRD
            ])

            spectral_mixup_XAFS = spectral_mixup(spectral_mixup_XAFS, n_spec=5, alpha=0.01, quantity=1)
            spectral_mixup_XRD = spectral_mixup(spectral_mixup_XRD, n_spec=5, alpha=0.3, quantity=1)

            New_Xtest = np.zeros((len(spectral_mixup_XAFS), 2400))
            augmented_data_list = []
            counter = 0

            for index, (xafs, xrd) in enumerate(zip(spectral_mixup_XAFS, spectral_mixup_XRD)):
                if index == New_Xtest.shape[0]:
                    break
                # Concatenate XAFS + XRD + the remaining original features
                augmented_data = np.concatenate((xafs, xrd, X_train[counter, 600:]), axis=0)
                New_Xtest[index] = np.round(augmented_data, 5)
                augmented_data_list.append(augmented_data)
                counter = (counter + 1) % len(X_train)

            # 2️⃣ Predict and build DataFrame rows
            results_rows = []

            for model in model_data_list:
                if model['model'].__class__.__name__ == 'LinearRegression':
                    print(f"Model: {type(model['model']).__name__}")
                    predictedscore1 = model['model'].predict(New_Xtest)

                    for index, (score, aug_data) in enumerate(zip(predictedscore1, augmented_data_list)):
                        # Build a dict with all columns
                        row = {
                            'Index': index,
                            'Predicted score': score
                        }
                        for i, val in enumerate(aug_data):
                            row[f'AugData_{i}'] = val
                        results_rows.append(row)

            # 3️⃣ Convert to DataFrame and save
            results_df = pd.DataFrame(results_rows)
            results_df.to_excel(f"{DIR_Data}/Aug_Predicted_Octa_MA.xlsx", index=False)



# 📂 Load your model
model_filename = f"{DIR_Data}/models_scores_MA/SingleDataSet_model_1_LinearRegression_MA.pkl"
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# 🟢 Define your slices here
slices = {
    "XAFS": (0, 300),
    "XRD": (300, 600),
    "Remaining1": (600, 900),
    "PDF": (900, 1200),
    "VB": (1200, 1500),
    "Remaining2": (1500, None)  # None means till end
}

desiredspectrum = 10000 // X_train.shape[0]

# Loop through each slice
for slice_name, (start, end) in slices.items():
    print(f"\n🟢 Processing slice: {slice_name}")
    file_ext=slice_name
    # Extract the slice
    if end is not None:
        spectra_data = X_train[:, start:end]
    else:
        spectra_data = X_train[:, start:]

    x = np.linspace(0, spectra_data.shape[1], num=spectra_data.shape[1]).reshape(-1, 1)
    sigma = 0.05
    gamma_val = 0.01

    K_rbf = rbf_kernel(x, gamma=gamma_val)

    # Generate GP-augmented data
    spectral_mixup_data = np.empty((0, 0))
    for data in spectra_data:
        gp_mean = gp_posterior_mean(K_rbf, data.reshape(-1, 1), sigma)[:, 0]
        gp_cov = gp_posterior_cov(K_rbf, sigma)
        gp_sample = np.random.multivariate_normal(gp_mean, gp_cov, desiredspectrum)
        if spectral_mixup_data.size == 0:
            spectral_mixup_data = gp_sample
        else:
            spectral_mixup_data = np.vstack((spectral_mixup_data, gp_sample))

    # Apply spectral mixup
    spectral_mixup_data = spectral_mixup(spectral_mixup_data, n_spec=5, alpha=0.01, quantity=1)

    # Prepare test array
    New_Xtest = np.zeros((len(spectral_mixup_data), spectra_data.shape[1]))
    augmented_data_list = []

    for index, aug in enumerate(spectral_mixup_data):
        if index == New_Xtest.shape[0]:
            break
        New_Xtest[index] = np.round(aug, 5)
        augmented_data_list.append(aug)

    # Predict
    predicted_scores = loaded_model.predict(New_Xtest)

    # Save logs
    log_file = f'{DIR_Data}/Aug_Predicted_data_{slice_name}_{file_ext}_MA.txt'
    csv_file = f'{DIR_Data}/predicted_scores_{slice_name}_{file_ext}_MA.csv'

    with open(log_file, 'w') as log, open(csv_file, 'w') as csv:
        log.write('Index,Predicted Score,Augmented Data\n')
        csv.write('Predicted Score\n')

        for idx, (score, aug_data) in enumerate(zip(predicted_scores, augmented_data_list)):
            log.write(f"{idx},{score},{','.join(map(str, aug_data))}\n")
            csv.write(f"{score}\n")

    print(f"✅ Saved logs for {slice_name} to:")
    print(f"   - {log_file}")
    print(f"   - {csv_file}")
