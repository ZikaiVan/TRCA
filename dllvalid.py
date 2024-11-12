# -*- coding: utf-8 -*-
import sys, csv
import pandas as pd
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets import WearableDataset_wet, WearableDataset_dry
from SSVEPAnalysisToolbox.utils.wearablepreprocess import preprocess, filterbank, suggested_ch, \
    suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms import (
    SCCA_qr, SCCA_canoncorr, ECCA, MSCCA, MsetCCA, MsetCCAwithR,
    TRCA, ETRCA, MSETRCA, MSCCA_and_MSETRCA, TRCAwithR, ETRCAwithR, SSCOR, ESSCOR,
    TDCA
)
from SSVEPAnalysisToolbox.evaluator import cal_acc, cal_itr
import time
import ctypes
import numpy as np

def write_4d_array_to_csv(array, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    writer.writerow(array[i, j, k, :])

def original():
    num_subbands = 5
    data_type = 'wet'

    # Prepare dataset
    if data_type.lower() == 'wet':
        dataset = WearableDataset_wet(path='Wearable')
    else:
        dataset = WearableDataset_dry(path='Wearable')
    dataset.regist_preprocess(preprocess)
    dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, num_subbands))

    # Prepare recognition model
    weights_filterbank = suggested_weights_filterbank(num_subbands, data_type, 'trca')
    recog_model = TRCA(weights_filterbank=weights_filterbank)

    # Set simulation parameters
    ch_used = suggested_ch()
    all_trials = [i for i in range(dataset.trial_num)]
    harmonic_num = 5
    tw = 2
    # test_block_idx = 8
    # test_block_list, train_block_list = dataset.leave_one_block_out(block_idx = test_block_idx)

    # Get training data and train the recognition model
    ref_sig = dataset.get_ref_sig(tw, harmonic_num)
    freqs = dataset.stim_info['freqs']
    # 这里get data是改过的数据截取范围(get_data_single_trial), 不能用原始toolbox来测, 用目录下的这个toolbox来测
    X_train, Y_train = dataset.get_data(sub_idx=sub_idx,
                                        blocks=train_block_list,
                                        trials=all_trials,
                                        channels=ch_used,
                                        sig_len=tw)
    a = np.array(X_train)
    recog_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs)
    a = np.array(recog_model.model['template_sig'])
    b = np.array(recog_model.model['U'])

    write_4d_array_to_csv(a, 'template_ori.csv')
    write_4d_array_to_csv(b, 'u_ori.csv')

    # Get testing data and test the recognition model
    X_test, Y_test = dataset.get_data(sub_idx=sub_idx,
                                    blocks=test_block_list,
                                    trials=all_trials,
                                    channels=ch_used,
                                    sig_len=tw)
    X_test = np.array(X_test)
    write_4d_array_to_csv(X_test, 'test_ori.csv')

    pred_label = recog_model.predict(X_test)
    acc = cal_acc(Y_true=Y_test, Y_pred=pred_label[0])

    return pred_label[0], acc

def dll():
    num_subbands = 5
    data_type = 'wet'

    # Prepare dataset
    if data_type.lower() == 'wet':
        dataset = WearableDataset_wet(path='Wearable')
    else:
        dataset = WearableDataset_dry(path='Wearable')
    dataset.preprocess_fun=None
    # Prepare recognition model
    weights_filterbank = suggested_weights_filterbank(num_subbands, data_type, 'trca')

    # Set simulation parameters
    ch_used = suggested_ch()
    all_trials = [i for i in range(dataset.trial_num)]
    harmonic_num = 5
    tw = 2

    # test_block_idx = 8
    # test_block_list, train_block_list = dataset.leave_one_block_out(block_idx = test_block_idx)

    # Get training data and train the recognition model
    ref_sig = dataset.get_ref_sig(tw, harmonic_num)
    freqs = dataset.stim_info['freqs']

    ###########################################    TRAIN    ###############################################
    X_train, Y_train = dataset.get_data(sub_idx=sub_idx,
                                        blocks=train_block_list,
                                        trials=all_trials,
                                        channels=ch_used,
                                        sig_len=tw)

    arr = np.array(X_train)
    X_train = arr.reshape((9, 12, 8, 500))
    dll = ctypes.cdll.LoadLibrary('./TRCA.dll')
    pX_train = X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    template = np.empty((12, 5, 8, 500), dtype=np.double) 
    U = np.empty((5, 12, 8, 1), dtype=np.double)
    dTemplate = template.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # double pointer Template
    dU = U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dll.TrcaTrain(pX_train, dTemplate, dU, 250, 5, 9, 12, 8, 500)
    ###########################################    TEST    ###############################################
    X_test, Y_test = dataset.get_data(sub_idx=sub_idx,
                                    blocks=test_block_list,
                                    trials=all_trials,
                                    channels=ch_used,
                                    sig_len=tw)
    arr = np.array(X_test)
    X_test = arr.reshape((1, 12, 8, 500))
    Pred = np.empty((12), dtype=int)

    pX_test = X_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dPred = Pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    dll.TrcaTest(pX_test, dTemplate, dU, dPred, 250, 5, 12, 8, 500)
    Pred = np.ctypeslib.as_array(ctypes.cast(dPred, ctypes.POINTER(ctypes.c_int)), Pred.shape)
    acc = cal_acc(Y_true=Y_test, Y_pred=Pred)

    return Pred, acc


RUN_TEST = 1
RUN_ORI = 1
RUN_BOTH = 0

if __name__ == '__main__':
    train_block_list = [i for i in range(0, 9)]
    test_block_list = [9]

    dllTime=0
    dllAcc=0
    oriTime=0
    oriAcc=0
    for sub_idx in range(54, 55):
        print(sub_idx)
        
        if RUN_TEST or RUN_BOTH:
            tic = time.time()
            Pred, Acc = dll()
            dllTime += time.time()-tic
            dllAcc += Acc
            print(f"DLL Pred:{Pred}")
            print(f"DLL Acc:{Acc}")

        if RUN_ORI or RUN_BOTH:
            tic = time.time()
            Pred, Acc = original()
            oriTime += time.time()-tic
            oriAcc += Acc
            print(f"Ori Pred:{Pred}")
            print(f"Ori Acc:{Acc}")

    print(f"DLL time: {dllTime/100}, DLL acc: {dllAcc/100}")
    print(f"Ori time: {oriTime/100}, Ori acc: {oriAcc/100}")


    
