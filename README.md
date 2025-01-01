## TRCA-cpp-src
- class Cheby1Filter: Design of bandpass, bandstop filters

- class Preprocess: notch, FilterBank, detrend, average, std calculation

- class TRCA: TRCA implementation

- utils: data, debug function
  
- for Chinese, vstudio默认gb2312编码, vs code默认utf8编码, 大部分含中文注释文件已经转换为utf8编码

## TRCA.dll
- For source code, refer to[dll.cpp](./dll.cpp).
  
- For python call, refer to[dllValid.py](./dllValid.py), and [onlineValid.py](./onlineValid.py)

### FilterBank
- Implementation of notch, FilterBank (10 banks,  6-th order bandpass at most: [{6, 14, 22, 30, 38, 46, 54, 62, 70, 78}, 90]), detrend, normalization.
  
- We use Cheby1 to design all filters.

- Input(2 pointers, 7 int)
    - darray: double* (row major 4D array, [block, stimulus, electrode, samples]). Generally, block=1, stimulus=the number of trials.

    - dout: double* (row major 4D array, [block*stimulus, subbands number, electrode, samples])

    - s_rate: int (sampling rate)

    - subbands: int (the number of subbands in FilterBank)

    - len: int (the number of blocks)

    - stimulus: int (the number of stimulus)

    - electrodes: int (the number of electrodes)

    - num_samples: int (the number of samples)

    - debug: int (=1 to debug mode, the data of FilterBank will be written into ./FilterBank.csv)

- Output
    - dout: memcpy to dout*

    - exit code (TODO)
### TrcaTrain
TRCA training (=FilterBank + TrcaTrainOnly)

- Input (3 pointers, 7 int)
    - darray: double* (row major 4D array, [train blocks, stimulus, electrodes, samples])

    - pTemplate: double* (4D array, [stimulus, subbans, electrodes, samples]). Generally, we don't consider it is row major or col major, since it will be directly used in TrcaTest. This function will output col major data, **The same applies hereinafter**.

    - pU: double* (4D array, [subbands, stimulus, electrodes, 1]). Generally, we don't consider it is row major or col major, since it will be directly used in TrcaTest. This function will output col major data, **The same applies hereinafter**.

    - s_rate: int (sampling rate)

    - subbands: int (the number of subbands in FilterBank)

    - train_len: int (the number of train blocks)

    - stimulus: int (the number of stimulus)

    - electrodes: int (the number of electrodes)

    - num_samples: int (the number of samples)

    - debug: int (=1 to csv_output mode, `templates`, and `u` will be written into ./\*.csv. =2 to debug mode, the data of FilterBank, `templates`, and `u` will be written into ./\*.csv.)

- Output
    - pTemplate: memcpy to pTemplate*

    - pU: memcpy to pU

    - exit code (TODO)

### TrcaTrainOnly
Only train of TRCA is implemented, without FilterBank.

- Input (3 pointers, 7int)
    - darray: double* (row major 4D array, [train blocks*stimulus, subbands, electrodes, samples]), which is the output of FilterBank

    - pTemplate: double* (4D array, [stimulus, subbands, electrodes, samples])

    - pU: double* (4D array, [subbands, stimulus, electrodes, 1])

    - s_rate: int (sampling rate)

    - subbands: int (the number of subbands)

    - train_len: int (the number of train blocks)

    - stimulus: int (the number of stimulus)

    - electrodes: int (the number of electrodes)

    - num_samples: int (the number of samples)

    - debug: int (=1 to csv_output mode, `templates`, and `u` will be written to ./\*.csv. =2 to debug mode, `input`, `templates`, and `u` will be written to ./\*.csv)

- Output
    - pTemplate: memcpy to pTemplate*

    - pU: memcpy to pU*

    - exit code (TODO)

### TrcaTest
TRCA testing (=FilterBank + TrcaTestOnly)

- Input (5 pointers, 7 int)
    - darray: double* (row major 4D array, [1, test blocks, electrodes, samples])

    - pTemplate: double* (output of TrcaTrain)

    - pU: double*, (output of TrcaTrain)

    - pcoeff: double* (1D array, [test blocks*stimulus])

    - pPred: int* (1D array, [test blocks])

    - s_rate: int (sampling rate)

    - subbands: int (the number of subbands)

    - test_len: int (the number of test blocks)

    - stimulus: int (the number of stimulus)

    - electrodes: int (the number of electrodes)

    - num_samples: int (the number of samples)

    - debug: int (=1 to debug mode, `templates`, and `u` will be written to ./\*.csv)

- Output
    - pPred: memcpy to

    - pcoeff: memcpy to

    - exit code (TODO)

### TrcaTestOnly
Only test of TRCA is implemented, without FilterBank.

- Input (5 pointers, 7 int)
    - darray: double* (row major 4D array, [1*test blocks, subbands, electrodes, samples])

    - pTemplate: double*, (output of TrcaTrain)

    - pU: double*, (output of TrcaTrain)

    - pcoeff: double* (1D array, [test blocks*stimulus])

    - pPred: int* (1D array, [test blocks])

    - s_rate: int (sampling rate)

    - subbands: int (the number of subbands)

    - test_len: int (the number of test blocks)

    - stimulus: int (the number of stimulus)

    - electrodes: int (the number of electrodes)

    - num_samples: int (the number of samples)

    - debug: int (=1 to debug mode, `templates`, `u` ,and `input` will be written to ./\*.csv)

- Output
    - pPred: memcpy to pPred*

    - pcoeff: memcpy to pcoeff*

    - exit code (TODO)

### TrcaTestCsv
TRCA testing (=FilterBank + TrcaTestOnly), which use template.csv and u.csv for input

- Input (5 pointers, 7 int)
    - darray: double* (row major 4D array, [1, test blocks, electrodes, samples])

    - pTemplate: char*, path to templates.csv

    - pU: char*, path to u.csv

    - pcoeff: double* (1D array, [test blocks*stimulus])

    - pPred: int* (1D array, [test blocks])

    - s_rate: int (sampling rate)

    - subbands: int (the number of subbands)

    - test_len: int (the number of test blocks)

    - stimulus: int (the number of stimulus)

    - electrodes: int (the number of electrodes)

    - num_samples: int (the number of samples)

    - debug: int (=1 to debug mode, `templates`, and `u` will be written to ./\*.csv)

- Output
    - pPred: memcpy to pPred*

    - pcoeff: memcpy to pcoeff*

    - exit code (TODO)

### TrcaTestOnlyCsv
TRCA testing, without FilterBank, which use template.csv and u.csv for input

- Input (5 pointers, 7 int)
    - darray: double* (row major 4D array, [1, test blocks, electrodes, samples])

    - pTemplate: char*, path to templates.csv

    - pU: char*, path to u.csv

    - pcoeff: double* (1D array, [test blocks*stimulus])

    - pPred: int* (1D array, [test blocks])

    - s_rate: int (sampling rate)

    - subbands: int (the number of subbands)

    - test_len: int (the number of test blocks)

    - stimulus: int (the number of stimulus)

    - electrodes: int (the number of electrodes)

    - num_samples: int (the number of samples)

    - debug: int (=1 to debug mode, `templates`, and `u` will be written to ./\*.csv)

- Output
    - pPred: memcpy to pPred*

    - pcoeff: memcpy to pcoeff*

    - exit code (TODO)


## Application
- dllvalid.py: use [SSVEP-AnaTool] (https://github.com/pikipity/SSVEP-Analysis-Toolbox) for testing. Need to check the version of toolbox, or use toolbox in this repo.

- Use Wearable-SSVEP (wet) dataset for testing. When using SSVEPAnalysisToolbox, it is better to check if `get_data` function in the toolbox can get correct data segments. Need to compare the toolbox code and the dataset documentation. 

- If data rearrange is needed, we suggest use loop to manually do that, rather than transpose or reshpae in third-party lib.
