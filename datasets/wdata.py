import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, Dataset


def interpolate(val, A, B, X, Y):
    val = float(val)
    A = float(A)
    B = float(B)
    X = float(X)
    Y = float(Y)
    out = float(((val - A) * (Y - X)) / (B - A) + X)
    return out

def extract_data(dataframe, header, addHeader, slice_start, slice_end):
    headerList = []
    for i in range(len(dataframe.index)):
        headerItem = dataframe[header][i]
        
        if header == "KG2":
            if headerItem[slice_start - 1] == "-":
                headerList.append(int(headerItem[slice_start:slice_end]) / -10)
            elif headerItem[slice_start - 1] == "+":
                headerList.append(int(headerItem[slice_start:slice_end]) / 10)
        else:
           headerList.append(int(headerItem[slice_start:slice_end])) 
            
    headerListdf = pd.DataFrame({addHeader: headerList})
    print(headerListdf)
    print("---------------------------------")
    dataframe = dataframe.join(headerListdf)
    return headerListdf


class WeatherDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("WeatherDataLoader")
        
        precipitationData = pd.concat([pd.read_csv('./data/2015INDweather.csv', usecols=['DATE', 'AA1'])])
        pressureData = pd.concat([pd.read_csv('./data/2015INDweather.csv', usecols=['DATE', 'MF1'])])
        humidityData = pd.concat([pd.read_csv('./data/2015INDweather.csv', usecols=['DATE', 'RH1'])])
        cloudData = pd.concat([pd.read_csv('./data/2015INDweather.csv', usecols=['DATE', 'GA1'])])

        # Precipitation Data
        precipitationData["AA1"] = precipitationData["AA1"].astype(str)
        precipitationData = precipitationData[precipitationData["AA1"].str.startswith("24")]
        precipitationData = precipitationData[precipitationData["DATE"].str.contains("04:59:00")]
        precipitationData = precipitationData.reset_index(drop=True)

        # Pressure Data
        pressureData = pressureData.dropna()
        pressureData["MF1"] = pressureData["MF1"].astype(str)
        pressureData = pressureData[pressureData["DATE"].str.contains("04:59:00")]
        pressureData = pressureData.reset_index(drop=True)

        # Humidity Data
        humidityData = humidityData.dropna()
        humidityData["RH1"] = humidityData["RH1"].astype(str)

        # Obtain Daily Averages
        humidityData = humidityData[humidityData["DATE"].str.contains("04:59:00")]
        humidityData = humidityData.reset_index(drop=True)

        #Sky Coverage Data (label data)
        cloudData = cloudData.dropna()
        cloudData = cloudData[cloudData["DATE"].str.contains("11:54:00")]
        cloudData = cloudData.reset_index(drop=True)




        # DATA EXTRACTION

        # Extract precipitation data
        precipitationHeader = "PRECIP (in mm)"
        precipitationData = extract_data(precipitationData, "AA1", precipitationHeader, 3, 7)

        # Extract pressure data
        pressureHeader = "PRESSURE(in hPa)"
        pressureData = extract_data(pressureData, "MF1", pressureHeader, 0, 5)

        # Extract humidity data
        humidityHeader = "REL HUMID (%)"
        humidityData = extract_data(humidityData, "RH1", humidityHeader, 6, 9)

        # Extract Cloud Cover data
        cloudHeader = "SKY COVERAGE"
        cloudData = extract_data(cloudData, "GA1", cloudHeader, 0, 2)

        for index, row in cloudData.iterrows():
            if(row['SKY COVERAGE'] < 3):
                row['SKY COVERAGE'] = 0
            else:
                row['SKY COVERAGE'] = 1
        
        print(cloudData)





        precipitationTensor = torch.tensor(precipitationData["PRECIP (in mm)"].values)
        precipTensTrain = torch.narrow(precipitationTensor, 0, 0, 56)
        precipTensValid = torch.narrow(precipitationTensor, 0, 56, 14)


        pressureTensor = torch.tensor(pressureData["PRESSURE(in hPa)"].values)
        pressTensTrain = torch.narrow(pressureTensor, 0, 0, 56)
        pressTensValid = torch.narrow(pressureTensor, 0, 56, 14)


        humidityTensor = torch.tensor(humidityData["REL HUMID (%)"].values)
        humidTensTrain = torch.narrow(humidityTensor, 0, 0, 56)
        humidTensValid = torch.narrow(humidityTensor, 0, 56, 14)

        cloudTensor = torch.tensor(cloudData["SKY COVERAGE"].values)
        labelTrain = torch.narrow(cloudTensor, 0, 0, 56)
        labelValid = torch.narrow(cloudTensor, 0, 56, 14)



        trainList = [precipTensTrain, pressTensTrain, humidTensTrain]
        trainTensor = torch.stack(trainList)
        #trainList = [trainTensor, labelTrain]
        #trainTensor = torch.stack(trainList)


        validList = [precipTensValid, pressTensValid, humidTensValid]
        validTensor = torch.stack(validList)
        #trainList = [validTensor, labelValid]
        #trainTensor = torch.stack(trainList)


        trainTensor = torch.t(trainTensor)
        validTensor = torch.t(validTensor)
        
        trainTensor.type(dtype=torch.float)
        validTensor.type(dtype=torch.float)
        
        trainMin0 = float(torch.min(trainTensor.select(1, 0)))
        trainMin1 = float(torch.min(trainTensor.select(1, 1)))
        trainMin2 = float(torch.min(trainTensor.select(1, 2)))
        trainMax0 = float(torch.max(trainTensor.select(1, 0)))
        trainMax1 = float(torch.max(trainTensor.select(1, 1)))
        trainMax2 = float(torch.max(trainTensor.select(1, 2)))
        
        validMin0 = float(torch.min(validTensor.select(1, 0)))
        validMin1 = float(torch.min(validTensor.select(1, 1)))
        validMin2 = float(torch.min(validTensor.select(1, 2)))
        validMax0 = float(torch.max(validTensor.select(1, 0)))
        validMax1 = float(torch.max(validTensor.select(1, 1)))
        validMax2 = float(torch.max(validTensor.select(1, 2)))
        
        min0 = trainMin0
        min1 = trainMin1
        min2 = trainMin2
        max0 = trainMax0
        max1 = trainMax1
        max2 = trainMax2
        print(max0)
        
        if trainMin0 < validMin0:
            min0 = trainMin0
        else:
            min0 = validMin0
            
        if trainMin1 < validMin1:
            min1 = trainMin1
        else:
            min1 = validMin1
            
        if trainMin2 < validMin2:
            min2 = trainMin2
        else:
            min2 = validMin2
            
        if trainMax0 > validMax0:
            max0 = trainMax0
        else:
            max0 = validMax0
            
        if trainMax1 > validMax1:
            max1 = trainMax1
        else:
            max1 = validMax1
            
        if trainMax2 > validMax2:
            max2 = trainMax2
        else:
            max2 = validMax2
            
        for i in range(trainTensor.size(0)):
            trainTensor[i, 0] = interpolate(trainTensor[i, 0], min0, max0, 0.0, 1.0)
            trainTensor[i, 1] = interpolate(trainTensor[i, 1], min1, max1, 0.0, 1.0)
            trainTensor[i, 2] = interpolate(trainTensor[i, 2], min2, max2, 0.0, 1.0)
            
        print(trainTensor.size())
        print(validTensor)
        print(labelTrain.size())
        print(labelValid.size())
        
        train_data = TensorDataset(trainTensor, labelTrain)
        valid_data = TensorDataset(validTensor, labelValid)
        
        self.train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_data, batch_size=self.config.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass



