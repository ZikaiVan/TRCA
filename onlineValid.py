import scipy.io
import numpy as np
import ctypes

file = 'S6_benchmark.mat'
mat_data = scipy.io.loadmat(file)
####################### Benchmark ##########################
# trials = np.transpose(numpy_data, (3, 2, 0, 1)) # 不能用这个
if file == 'S6_benchmark.mat':
    numpy_data = np.array(mat_data['data'])
    # 64, 1500, 40, 6
    trials = np.empty((6, 40, 64, 500))
    for i in range(0, numpy_data.shape[3]):
        for j in range(0, numpy_data.shape[2]):
            for k in range(0, numpy_data.shape[0]):
                trials[i, j, k, :] = numpy_data[k, 500:1000, j, i]

####################### 自采 #######################
if file == "s2_online_12_new_avg95.mat":
    numpy_data = np.array(mat_data['data'])
    # 8, 250, 8, 10
    trials = np.empty((10, 8, 8, 250))
    for i in range(0, numpy_data.shape[3]):
        for j in range(0, numpy_data.shape[2]):
            for k in range(0, numpy_data.shape[0]):
                trials[i, j, k, :] = numpy_data[k, :, j, i]

####################### 自采2 #######################
if file == "unity_trca_debug.mat":
    numpy_data = np.array(mat_data['data'])
    # 4, 5, 8, 500
    trials = np.empty((5, 4, 8, 500))
    for i in range(0, numpy_data.shape[1]):
        for j in range(0, numpy_data.shape[0]):
            for k in range(0, numpy_data.shape[2]):
                trials[i, j, k, :] = numpy_data[j, i, k, :]

stimulus = trials.shape[1]
electrodes = trials.shape[2]
num_samples = trials.shape[3]
subbands = 5
s_rate = 250
train_len = trials.shape[0] - 1

train = trials[0:train_len,:,:,:]
test = trials[train_len,:,:,:].reshape((1, -1, electrodes, num_samples))

dll = ctypes.cdll.LoadLibrary('./x64/Release/TRCA.dll')
template = np.empty((stimulus, subbands, electrodes, num_samples), dtype=np.double) 
u = np.empty((subbands, stimulus, electrodes, 1), dtype=np.double)
train_fb = np.empty((9*stimulus, subbands, electrodes, num_samples))

ptrain = train.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ptemplate = template.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
pu = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ptrain_fb = train_fb.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

###########################################    TRAIN    ###############################################
SPLIT_API = 0
DEBUG = 0
FROM_CSV = 0

if SPLIT_API:
    print('train split')
    dll.FilterBank(ptrain, ptrain_fb, 
                   s_rate, subbands, train_len, 
                   stimulus, electrodes, num_samples, DEBUG)
    dll.TrcaTrainOnly(ptrain_fb, ptemplate, pu, 
                      s_rate, subbands, train_len, stimulus, 
                      electrodes, num_samples, FROM_CSV + DEBUG)
else:
    print('train combined')
    dll.TrcaTrain(ptrain, ptemplate, pu, 
                  s_rate, subbands, train_len, stimulus, 
                  electrodes, num_samples, FROM_CSV + DEBUG)

###########################################    TEST    ###############################################
ans = []
ans_coeff = []
for i in range(0, test.shape[1]):
    single_test = test[:, i, :, :]
    pred = np.empty((1), dtype=int)
    coeff = np.empty((stimulus), dtype=np.double)
    test_fb = np.empty((1, subbands, electrodes, num_samples))

    ptest = single_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ppred = pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    pcoeff = coeff.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptest_fb = test_fb.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if FROM_CSV:
        ptemplate = ctypes.c_char_p(b"./templates_dll.csv")
        pu = ctypes.c_char_p(b"./u_dll.csv")

    if SPLIT_API:
        dll.FilterBank(ptest, ptest_fb, 
                       s_rate, subbands, 1, 1, 
                       electrodes, num_samples, DEBUG)
        if FROM_CSV:
            dll.TrcaTestOnlyCsv(ptest_fb, ptemplate, pu, pcoeff, ppred,
                            s_rate, subbands, 1, stimulus, 
                            electrodes, num_samples, DEBUG)
        else:
            dll.TrcaTestOnly(ptest_fb, ptemplate, pu, pcoeff, ppred,
                            s_rate, subbands, 1, stimulus, 
                            electrodes, num_samples, DEBUG)
    elif FROM_CSV:
        dll.TrcaTestCsv(ptest, ptemplate, pu, pcoeff, ppred, 
                        s_rate, subbands, 1, stimulus, 
                        electrodes, num_samples, DEBUG)
    else:
        dll.TrcaTest(ptest, ptemplate, pu, pcoeff, ppred, 
                     s_rate, subbands, 1, stimulus, 
                     electrodes, num_samples, DEBUG)
    pred = np.ctypeslib.as_array(ctypes.cast(ppred, ctypes.POINTER(ctypes.c_int)), pred.shape)
    coeff = np.ctypeslib.as_array(ctypes.cast(pcoeff, ctypes.POINTER(ctypes.c_double)), coeff.shape)
    ans.append(pred[0])
    ans_coeff.append(coeff)
print(ans)
