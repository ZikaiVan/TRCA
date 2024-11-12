## TRCA-cpp-src
- Cheby1Filter类: 带通、带阻滤波器设计
- Preprocess类: notch滤波、filterBank、detrend、均值计算、标准差计算
- TRCA类: TRCA算法实现
- utils: 包含数据函数、调试函数，其中一部分数据函数应该放到别的类里面
- vstudio默认gb2312编码, vs code默认utf8编码, 大部分含中文注释代码的文件已经转换为utf8编码

## TRCA-dll
- 具体实现参照[dll.cpp](./dll.cpp), python调用参照[dllvalid.py](./dllvalid.py)
### TrcaTrain
- 输入
    - darray: double指针(行优先的4D数组, [训练轮数, 刺激数, 电极数, 信号点数])
    - pTemplate: double指针(4D数组, [刺激数, filterBank数, 电极数, 信号点数])
    - pU: double指针(4D数组, [filterBank数量, 刺激数, 电极数, 1])
    - s_rate: int(采样率)
    - subbands: int(filterBank数量)
    - train_len: int(训练轮数)
    - stimulus: int(刺激数)
    - electrodes: int(电极数)
    - num_samples: int(信号点数)
- 输出
    - 计算得到的template和U通过memcpy的方式copy到pTemplate和pU地址上
    - 返回错误码(还没做)
### TrcaTest
- 输入
    - darray: double指针(行优先的3D数组, [测试次数, 电极数, 信号点数])
    - pTemplate: train得到的指针
    - pU: train得到的指针
    - pPred: int指针(1D数组, [测试次数])
    - s_rate: 同上
    - subbands: 同上
    - stimulus: 同上
    - electrodes: 同上
    - num_samples: 同上
- 输出
    - 计算得到的标签通过memcpy方式拷贝到pPred
    - 返回错误码(还没做)

## 测试
- dllvalid.py: 使用[SSVEP-AnaTool](https://github.com/pikipity/SSVEP-Analysis-Toolbox)测试, 需确认lib版本, 也可使用本repo中提供的lib
- 使用Wearable-SSVEP(wet)数据集测试, 其中使用SSVEPAnalysisToolbox库测试时, 需要确认库get_data方法截取的数据是否正确, 需比照lib代码和数据集说明