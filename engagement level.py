#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import time
import math
import datetime
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from model_classes import LatentFactorModel, FPMC, FPMCWT,SPLFM, RBMCWT,RBMC
from itertools import cycle

MAX_ITER = 1000
PRINT_ITER = MAX_ITER//10

UPPER_LIMIT = 100000 #Test batch upper bound
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 1 - TRAIN_RATIO - VALID_RATIO
random.seed(2041)
np.random.seed(2041)
def preprocess():
    data = pd.read_csv('2016_12.csv')
    viewer_id = data['viewer_id']
    ad_airing_id = data['ad_airing_id']
    hh_id = data['hh_id']
    # active_ratio = data['active_to_duration_ratio']
    # engagement level: attetion_to_duration_ratio
    attention_ratio = data['attention_to_duration_ratio']
    industry = data['industry_name']
    timeString = data['viewing_start_time_utc'].tolist()
    view_date = [] # y-m-d date format
    time_unix = [] # uniform time format by seconds
    for t in timeString:
        date_time_obj = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
        v_d = date_time_obj.date()
        t_u = time.mktime(date_time_obj.timetuple())
        time_unix.append(t_u)
        view_date.append(v_d)

    dictionary = {'viewer_id': viewer_id,'ads_id':ad_airing_id,'viewing_start_time_utc': timeString,
                  'time_unix': time_unix,'view_date':view_date,'attention_to_duration_ratio': attention_ratio,'households':hh_id, 'industry': industry}
    return dictionary
if os.path.exists('dt.pickle'):
    dt = pd.read_pickle('dt.pickle')
else:
    dictionary = preprocess()
    dt = pd.DataFrame(dictionary)
    dt.to_pickle('dt.pickle')

user_list = pd.unique(dt['viewer_id'])
userIDs = dict(zip(user_list,range(len(user_list))))
item_list = pd.unique(dt['ads_id'])
itemIDs = dict(zip(item_list,range(len(item_list))))
time_list = pd.unique(dt['time_unix'])
timeIDs = dict(zip(time_list,range(len(time_list))))
house_list = pd.unique(dt['households'])
houseIDs = dict(zip(house_list,range(len(house_list))))
industry_list = pd.unique(dt['industry'])
industryIDs = dict(zip(industry_list,range(len(industry_list))))
weekday_list = range(7)
weekdayIDs = dict(zip(weekday_list,range(len(weekday_list))))


interactionsEX = []
interactionsEXPerUser = defaultdict(list)
item_dict = dt.values.tolist()

if os.path.exists("interactionsEX.pickle") and os.path.exists("interactionsEXPerUser.pickle"):
    with open("interactionsEX.pickle", "rb") as fp:
        interactionsEX = pickle.load(fp)
    with open("interactionsEXPerUser.pickle", "rb") as fp:
        interactionsEXPerUser = pickle.load(fp)
else:
    for d in item_dict:
        u = d[0]
        i = d[1]
        t = d[3]
        # Append weekday to features
        wkd= d[4].weekday()
        r = d[5]
        hh = d[6]
        industry = d[7]
        # Dataset changing must check
        interactionsEX.append((t,u,i,hh,industry,wkd,r)) #6 with households(hh) industry (ind)
        interactionsEXPerUser[u].append((t,i,hh,industry,wkd,r))
    #Sort both
    interactionsEX.sort()
    for each_u in interactionsEXPerUser:
        interactionsEXPerUser[each_u].sort()
    with open("interactionsEX.pickle", "wb") as fp:
        pickle.dump(interactionsEX, fp)
    with open("interactionsEXPerUser.pickle", "wb") as fp:
        pickle.dump(interactionsEXPerUser, fp)



itemIDs['dummy'] = len(itemIDs)
items = list(itemIDs.keys())

#
def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

# Construct a trainset / testset with previous ratio, households, industry ...
# Note: Both sets have been fixed with load/store
if os.path.exists("interactionsEXTrain.pickle") and \
    os.path.exists("interactionsEXValid.pickle") and \
    os.path.exists("interactionsEXTest.pickle"):
    with open("interactionsEXTrain.pickle", "rb") as fp:
        interactionsEXTrain = pickle.load(fp)
    with open("interactionsEXValid.pickle", "rb") as fp:
        interactionsEXValid = pickle.load(fp)
    with open("interactionsEXTest.pickle", "rb") as fp:
        interactionsEXTest = pickle.load(fp)
else:
    interactionsEXTrain =[]
    interactionsEXValid = []
    interactionsEXTest = []
    train_ratio = TRAIN_RATIO
    valid_ratio = VALID_RATIO
    test_ratio = TEST_RATIO
    for each_u in interactionsEXPerUser:
        this_data = interactionsEXPerUser[each_u]
        ui_len = len(this_data)
        primitive_list = []
        if ui_len<4: continue
        lastItem = 'dummy'
        # train_len = ui_len -1
        ratio_list = [d[-1] for d in this_data]
        lastRatio = sum(ratio_list) / len(ratio_list)
        for idx in range(ui_len):
            # Dataset changing must check
            t, i, hh, industry, wkd, r = interactionsEXPerUser[each_u][idx]
            if r > 0 and r <= 1: primitive_list.append((t,each_u,i,lastItem,lastRatio,hh,industry,wkd,r))
            lastItem = i
            lastRatio = r
        random.shuffle(primitive_list)
        train_len = round(len(primitive_list) * train_ratio)
        interactionsEXTrain += primitive_list[:train_len]
        valid_len = round(len(primitive_list) * valid_ratio)
        interactionsEXValid += primitive_list[train_len:train_len + valid_len]
        test_len = round(len(primitive_list) * test_ratio)
        interactionsEXTest+= primitive_list[train_len+valid_len:]
    with open("interactionsEXTrain.pickle", "wb") as fp:
        pickle.dump(interactionsEXTrain, fp)
    with open("interactionsEXValid.pickle", "wb") as fp:
        pickle.dump(interactionsEXValid, fp)
    with open("interactionsEXTest.pickle", "wb") as fp:
        pickle.dump(interactionsEXTest, fp)

mu = sum([each[-1] for each in interactionsEXTrain]) / len(interactionsEXTrain)
NSAMPLES = len(interactionsEXTrain) // 10
label_train = [d[-1] for d in interactionsEXTrain]
label_valid = [d[-1] for d in interactionsEXValid]

mse_train_trivial = MSE([mu]*len(label_train),label_train)
mse_valid_trivial = MSE([mu] * len(label_valid), label_valid)

#Test a trivial prediction
print("Trivial: " +"MSE_TRAIN: " + str(mse_train_trivial) + " MSE_Valid: " + str(mse_valid_trivial))

import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(2e-3)

# Iterator must be defined outside, otherwise it will be restarted everytime I call trainingStep
train_cycle = cycle(interactionsEXTrain)
def trainingStep(model, mode, interactions):

    #tag_t for time features tag_j for previous item features, tag_pr for previous rate features
    Nsamples = NSAMPLES
    with tf.GradientTape() as tape:
        sampleT, sampleU, sampleI, sampleJ, samplePR, sampleR = [], [], [], [], [], []
        sampleHH, sampleIND = [], []
        for _ in range(Nsamples):
            # Dataset changing must check
            t,u,i,j,pr,hh,industry,wkd,r = next(train_cycle)
            sampleT.append(timeIDs[t])
            # sampleT.append(weekdayIDs[wkd])
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])
            sampleHH.append(houseIDs[hh])
            # Very tricky handling of 'nan' industry
            if type(industry) is float:
                if math.isnan(industry):
                    sampleIND.append(2)
            else:
                sampleIND.append(industryIDs[industry])
            samplePR.append(pr)
            sampleR.append(r)
        if mode == 'LFM':
            #LFM
            loss = model(sampleU,sampleI,sampleR)
        elif mode == 'RBLFM':
            #RBLFM
            loss = model (sampleU,sampleI,samplePR,sampleR)
        elif mode == 'HBLFM':
            #HBLFM
            loss = model (sampleU,sampleI,sampleHH,sampleR)
        elif mode == 'IBLFM':
            #IBLFM
            loss = model (sampleU,sampleI,sampleIND,sampleR)
        elif mode == 'FPMC' or mode == 'FMC':
            #FPMC FMC
            loss = model(sampleU, sampleI, sampleJ, sampleR)
        elif mode == 'RBMC':
            #RBMC
            loss = model(sampleU, sampleI,sampleJ,samplePR,sampleR)
        elif mode == 'FPMCWT':
            #FPMCWT
            loss = model(sampleT, sampleU, sampleI, sampleJ, sampleR)
        elif mode == 'RBMCWT':
            #RBMCWT
            loss = model(sampleT, sampleU, sampleI, sampleJ, samplePR, sampleR)
        else:
            print('Undocumetned!')
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()

def TestStep(model, mode, interactions=[]):
    int_cyc = cycle(interactions)
    #tag_t for time features tag_j for previous item features, tag_pr for previous rate features
    pred = []
    upper_limit = UPPER_LIMIT
    with tf.GradientTape() as tape:
        it_size = len(interactions)
        batch_size = min(upper_limit,it_size)
        for b_idx in range(it_size//batch_size + 1):
            sampleT, sampleU, sampleI, sampleJ, samplePR= [], [], [], [], []
            sampleHH, sampleIND = [], []
            this_batch_size = min(batch_size, it_size - b_idx * batch_size)
            for idx in range(this_batch_size):
                # Dataset changing must check
                t,u,i,j,pr,hh,industry,wkd,r = next(int_cyc)
                sampleT.append(timeIDs[t])
                # sampleT.append(weekdayIDs[wkd])
                sampleU.append(userIDs[u])
                sampleI.append(itemIDs[i])
                sampleJ.append(itemIDs[j])
                sampleHH.append(houseIDs[hh])
                # Very tricky handling of 'nan' industry
                if type(industry) is float:
                    if math.isnan(industry):
                        sampleIND.append(2)
                else:
                    sampleIND.append(industryIDs[industry])
                samplePR.append(pr)
            if mode == 'LFM':
                #LFM
                this_pred = model.predictSample(sampleU,sampleI)
            elif mode == 'RBLFM':
                #RBLFM
                this_pred = model.predictSample(sampleU,sampleI,samplePR)
            elif mode == 'HBLFM':
                #HBLFM
                this_pred = model.predictSample(sampleU,sampleI,sampleHH)
            elif mode == 'IBLFM':
                #IBLFM
                this_pred = model.predictSample(sampleU,sampleI,sampleIND)
            elif mode == 'FPMC' or mode == 'FMC':
                #FPMC
                this_pred = model.predictSample(sampleU, sampleI, sampleJ)
            elif mode == 'RBMC':
                #RBMC
                this_pred = model.predictSample(sampleU, sampleI,sampleJ,samplePR)
            elif mode == 'FPMCWT':
                #FPMCWT
                this_pred = model.predictSample(sampleT, sampleU, sampleI, sampleJ)
            elif mode == 'RBMCWT':
                #RBMCWT
                this_pred = model.predictSample(sampleT, sampleU, sampleI, sampleJ, samplePR)
            else:
                print('Undocumetned!')
            pred += this_pred.numpy().tolist()
    return pred

# LFM
label_test = [d[-1] for d in interactionsEXTest]
modelLFM = LatentFactorModel(mu, 5, 0.00001,itemIDs,userIDs)

for i in range(MAX_ITER):
    obj = trainingStep(modelLFM,'LFM', interactionsEXTrain)
    if (i % PRINT_ITER == PRINT_ITER-1): print("iteration " + str(i+1) + ", objective = " + str(obj))


pred_train = TestStep(modelLFM,'LFM', interactionsEXTrain)
pred_valid = TestStep(modelLFM, 'LFM', interactionsEXValid)
pred_test = TestStep(modelLFM, 'LFM', interactionsEXTest)

mse_train = MSE(pred_train, label_train)
mse_valid = MSE(pred_valid, label_valid)
mse_test = MSE(pred_test,label_test)

print("LFM MSE_Train: " + str(mse_train) +" MSE_Valid: " + str(mse_valid))
print('LFM'+ str(mse_test))

#
#
# # Try to use PR (the previous attention ratio) instead of previous consumed ad
# modelRBLFM = SPLFM(mu, 5, 1e-5, userIDs, itemIDs) #Non-personalized with rating
# model,name = (modelRBLFM, "Ratio Based LFM")
# for i in range(MAX_ITER):
#     obj = trainingStep(model,'RBLFM', interactionsEXTrain)
#     if (i % PRINT_ITER == PRINT_ITER-1): print("iteration " + str(i+1) + ", objective = " + str(obj))
#
# pred_train = TestStep(modelRBLFM,'RBLFM', interactionsEXTrain)
# pred_valid = TestStep(modelRBLFM, 'RBLFM', interactionsEXValid)
#
#
# mse_train = MSE(pred_train, label_train)
# mse_valid = MSE(pred_valid, label_valid)
#
# print("RBLFM " +" MSE_Train: " + str(mse_train) + " MSE_Valid: " + str(mse_valid))

# Another LFM, but based on Household
# modelHBLFM = SPLFM(mu, 5, 1e-5, userIDs, itemIDs,linear = 0, fIDs=houseIDs) #Non-personalized with rating
# model,name = (modelHBLFM, "Households Based LFM")
# for i in range(MAX_ITER):
#     obj = trainingStep(model,'HBLFM', interactionsEXTrain)
#     if (i % PRINT_ITER == PRINT_ITER-1): print("iteration " + str(i+1) + ", objective = " + str(obj))
#
# pred_train = TestStep(modelHBLFM,'HBLFM', interactionsEXTrain)
# pred_valid = TestStep(modelHBLFM, 'HBLFM', interactionsEXValid)
#
#
# mse_train = MSE(pred_train, label_train)
# mse_valid = MSE(pred_valid, label_valid)
#
# print("HBLFM " +" MSE_Train: " + str(mse_train) + " MSE_Valid: " + str(mse_valid))
#
# # Another LFM Based on Industry
# modelIBLFM = SPLFM(mu, 5, 1e-5, userIDs, itemIDs,linear = 0, fIDs=industryIDs) #Non-personalized with rating
# model,name = (modelIBLFM, "Households Based LFM")
# for i in range(MAX_ITER):
#     obj = trainingStep(model,'IBLFM', interactionsEXTrain)
#     if (i % PRINT_ITER == PRINT_ITER-1): print("iteration " + str(i+1) + ", objective = " + str(obj))
#
# pred_train = TestStep(modelIBLFM,'IBLFM', interactionsEXTrain)
# pred_valid = TestStep(modelIBLFM, 'IBLFM', interactionsEXValid)
#
#
# mse_train = MSE(pred_train, label_train)
# mse_valid = MSE(pred_valid, label_valid)
#
# print("IBLFM " +" MSE_Train: " + str(mse_train) + " MSE_Valid: " + str(mse_valid))
# # RBMC
# modelRBMC = RBMC(mu,5,1e-5,1,1,userIDs,itemIDs)
#
# for i in range(MAX_ITER):
#     obj = trainingStep(modelRBMC,'RBMC', interactionsEXTrain)
#     if (i % PRINT_ITER == PRINT_ITER-1): print("iteration " + str(i+1) + ", objective = " + str(obj))
#
# pred_train = TestStep(modelRBMC,'RBMC', interactionsEXTrain)
# pred_valid = TestStep(modelRBMC, 'RBMC', interactionsEXValid)
#
# mse_train = MSE(pred_train, label_train)
# mse_valid = MSE(pred_valid, label_valid)
#
# print("RBMC " +" MSE_Train: " + str(mse_train) + " MSE_Valid: " + str(mse_valid))
# FPMCWT
# moodelFPMCWT = FPMCWT(mu,5,1e-5,1,1,timeIDs,userIDs,itemIDs)
#
# for i in range(MAX_ITER):
#     obj = trainingStep(moodelFPMCWT,'FPMCWT', interactionsEXTrain)
#     if (i % PRINT_ITER == PRINT_ITER-1): print("iteration " + str(i+1) + ", objective = " + str(obj))
#
# pred_train = TestStep(moodelFPMCWT,'FPMCWT', interactionsEXTrain)
# pred_valid = TestStep(moodelFPMCWT, 'FPMCWT', interactionsEXValid)
#
# mse_train = MSE(pred_train, label_train)
# mse_valid = MSE(pred_valid, label_valid)
#
# print("FPMCWT " +" MSE_Train: " + str(mse_train) + " MSE_Valid: " + str(mse_valid))
#
# # RBMCWT

# moodelRBMCWT = RBMCWT(mu,5,1e-5,1,1,timeIDs,userIDs,itemIDs)
# min_obj = 1
# for i in range(MAX_ITER):
#
#     obj = trainingStep(moodelRBMCWT, 'RBMCWT', interactionsEXTrain)
#     if min_obj > obj: min_obj = obj
#     if (i % PRINT_ITER == PRINT_ITER-1):
#         print("iteration " + str(i+1) + ", objective = " + str(obj))
#         pred_valid = TestStep(moodelRBMCWT, 'RBMCWT', interactionsEXValid)
#         mse_valid = MSE(pred_valid, label_valid)
#         print("iteration " + str(i+1) + ' mse_valid: '+str(mse_valid))
#
# # tf.saved_model.save(moodelRBMCWT,'FPMC_TR')
# pred_valid = TestStep(moodelRBMCWT, 'RBMCWT', interactionsEXValid)
# pred_train = TestStep(moodelRBMCWT, 'RBMCWT', interactionsEXTrain)
# pred_test = TestStep(moodelRBMCWT, 'RBMCWT', interactionsEXTest)
#
# mse_train = MSE(pred_train, label_train)
# mse_valid = MSE(pred_valid, label_valid)
# mse_test = MSE(pred_test, label_test)
# print("RBMCWT " +" MSE_Train: " + str(mse_train) + " MSE_Valid: " + str(mse_valid))
# print("RBMCWT " + "MSE_TEST:" + str(mse_test))

# # Let's try complicated models
# # 2 kinds of FPMC
# modelFMC = FPMC(mu, 5, 1e-5, 0, 1, userIDs, itemIDs) #Non-personalized
# modelFPMC = FPMC(mu, 5, 1e-5, 1, 1, userIDs, itemIDs)
#
# for model,name,mode in [(modelFMC,"FMC",'FMC'),(modelFPMC,"FPMC",'FPMC')]:
#     for i in range(MAX_ITER):
#         obj = trainingStep(model, mode, interactionsEXTrain)
#         if (i % PRINT_ITER == PRINT_ITER-1): print("iteration " + str(i+1) + ", objective = " + str(obj))
#
#     pred_train = TestStep(model, mode, interactionsEXTrain)
#     pred_valid = TestStep(model, mode, interactionsEXValid)
#     mse_train = MSE(pred_train, label_train)
#     mse_valid = MSE(pred_valid, label_valid)
#
#     print(name + " MSE_Train: " + str(mse_train) + " MSE_Valid: " + str(mse_valid))