## TRCA-cpp-src
- Cheby1Filter类: 带通、带阻滤波器设计

- Preprocess类: notch滤波、filterBank、detrend、均值计算、标准差计算

- TRCA类: TRCA算法实现

- utils: 包含数据函数、调试函数
    - TODO: 其中一部分数据函数应该放到别的类里面

- vstudio默认gb2312编码，vs code默认utf8编码，大部分含中文注释代码的文件已经转换为utf8编码

## TRCA.dll
- 具体实现参照[dll.cpp](./dll.cpp)，python调用参照[dllValid.py](./dllValid.py)、[onlineValid.py](./onlineValid.py)

### FilterBank
完成notch滤波、filterBank滤波（最多10组6阶带通，[{6, 14, 22, 30, 38, 46, 54, 62, 70, 78}，90]）、去直流、标准化操作，滤波器设计为Cheby1

- 输入（2指针、7int）
    - darray: double指针（行优先的4D数组，[轮数，刺激数，电极数，信号点数]），一般情况下，轮数=1，刺激数=trials数即可

    - dout: double指针（行优先的4D数组，[轮数*刺激数，filterBank数，电极数，信号点数]）

    - s_rate: int（采样率）

    - subbands: int（filterBank数量）

    - len: int（轮数）

    - stimulus: int（刺激数）

    - electrodes: int（电极数）

    - num_samples: int（信号点数）

    - debug: int，传入1进入调试，将filterBank数据覆盖写入路径下csv文件

- 输出
    - dout: 指针拷贝数据

    - 返回错误码（还没做，下同）

### TrcaTrain
完整的TRCA训练流程，filterBank+TrcaTrainOnly

- 输入（3指针、7int）
    - darray: double指针（行优先的4D数组，[训练轮数，刺激数，电极数，信号点数]）

    - pTemplate: double指针（4D数组，[刺激数，filterBank数，电极数，信号点数]），一般不考虑该数组的存储主序，因为会直接输入给Test使用，当前输出的是列优先的数据，**下同**

    - pU: double指针（4D数组，[filterBank数量，刺激数，电极数，1]），一般不考虑该数组的存储主序，因为会直接输入给Test使用，当前输出的是列优先的数据，**下同**

    - s_rate: int（采样率）

    - subbands: int（filterBank数量）

    - train_len: int（训练轮数）

    - stimulus: int（刺激数）

    - electrodes: int（电极数）

    - num_samples: int（信号点数）

    - debug: int，传入1时将templates和u覆盖写入路径下csv文件，传入2时将input、filterBank数据、templates和u覆盖写入路径下csv文件

- 输出
    - pTemplate: 指针拷贝数据

    - pU: 指针拷贝数据

    - 返回错误码


### TrcaTrainOnly
只执行Trca训练功能

- 输入（3指针、7int）
    - darray: double指针（行优先的4D数组，[训练轮数*刺激数，filterBank数，电极数，信号点数]），即filterBank完成后的数据

    - pTemplate: double指针（4D数组，[刺激数，filterBank数，电极数，信号点数]）

    - pU: double指针（4D数组，[filterBank数量，刺激数，电极数，1]）

    - s_rate: int（采样率）

    - subbands: int（filterBank数量）

    - train_len: int（训练轮数）

    - stimulus: int（刺激数）

    - electrodes: int（电极数）

    - num_samples: int（信号点数）

    - debug: int，传入1时将templates和u覆盖写入路径下csv文件，传入2时将input、filterBank数据、templates和u覆盖写入路径下csv文件

- 输出
    - pTemplate: 指针拷贝数据

    - pU: 指针拷贝数据

    - 返回错误码

### TrcaTest
完整的TRCA测试流程，filterBank+TrcaTestOnly

- 输入（5指针、7int）
    - darray: double指针（行优先的4D数组，[1，测试次数，电极数，信号点数]）

    - pTemplate: double指针，train得到的指针

    - pU: double指针，train得到的指针

    - pcoeff: double指针（1D数组，[测试次数*stimulus]）

    - pPred: int指针（1D数组，[测试次数]）

    - s_rate: int（采样率）

    - subbands: int（filterBank数量）

    - test_len: int（训练轮数）

    - stimulus: int（刺激数）

    - electrodes: int（电极数）

    - num_samples: int（信号点数）

    - debug: int，传入1时进入调试，将templates、u和filterBank数据覆盖写入路径下csv文件

- 输出
    - pPred: 指针拷贝数据

    - pcoeff: 指针拷贝数据

    - 返回错误码

### TrcaTestOnly
只执行Trca测试功能

- 输入（5指针、7int）
    - darray: double指针（行优先的4D数组，[1*测试次数，filterBank数，电极数，信号点数]）

    - pTemplate: double指针，train得到的指针

    - pU: double指针，train得到的指针

    - pcoeff: double指针（1D数组，[测试次数*stimulus]）

    - pPred: int指针（1D数组，[测试次数]）

    - s_rate: int（采样率）

    - subbands: int（filterBank数量）

    - test_len: int（训练轮数）

    - stimulus: int（刺激数）

    - electrodes: int（电极数）

    - num_samples: int（信号点数）

    - debug: int，传入1时进入调试，将templates、u和输入数据覆盖写入路径下csv文件

- 输出
    - pPred: 指针拷贝数据

    - pcoeff: 指针拷贝数据

    - 返回错误码

### TrcaTestCsv
完整的TRCA测试流程，filterBank+TrcaTestOnly，使用csv文件输入template和u

- 输入（5指针、7int）
    - darray: double指针（行优先的4D数组，[1，测试次数，电极数，信号点数]）

    - pTemplate: char指针，存放templates的csv文件路径

    - pU: char指针，存放u的csv文件路径

    - pcoeff: double指针（1D数组，[测试次数*stimulus]）

    - pPred: int指针（1D数组，[测试次数]）

    - s_rate: int（采样率）

    - subbands: int（filterBank数量）

    - test_len: int（训练轮数）

    - stimulus: int（刺激数）

    - electrodes: int（电极数）

    - num_samples: int（信号点数）

    - debug: int，传入1时进入调试，将templates、u和filterBank数据覆盖写入路径下csv文件

- 输出
    - pPred: 指针拷贝数据

    - pcoeff: 指针拷贝数据

    - 返回错误码

### TrcaTestOnlyCsv
只执行Trca测试功能，使用csv文件输入template和u

- 输入（5指针、7int）
    - darray: double指针（行优先的4D数组，[1，测试次数，电极数，信号点数]）

    - pTemplate: char指针，存放templates的csv文件路径

    - pU: char指针，存放u的csv文件路径

    - pcoeff: double指针（1D数组，[测试次数*stimulus]）

    - pPred: int指针（1D数组，[测试次数]）

    - s_rate: int（采样率）

    - subbands: int（filterBank数量）

    - test_len: int（训练轮数）

    - stimulus: int（刺激数）

    - electrodes: int（电极数）

    - num_samples: int（信号点数）

    - debug: int，传入1时进入调试，将templates、u和filterBank数据覆盖写入路径下csv文件

- 输出
    - pPred: 指针拷贝数据

    - pcoeff: 指针拷贝数据

    - 返回错误码


## 测试
- dllvalid.py: 使用[SSVEP-AnaTool]（https://github.com/pikipity/SSVEP-Analysis-Toolbox）测试，需确认toolbox版本，也可使用本repo中提供的toolbox

- 使用Wearable-SSVEP（wet）数据集测试，其中使用SSVEPAnalysisToolbox库测试时，需要确认库get_data方法截取的数据是否正确，需比照toolbox代码和数据集说明

- 测试数据重排请使用可控的循环实现，避免调用transpose等api