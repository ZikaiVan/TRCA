### TRCA-cpp
- Cheby1Filter类: 带通、带阻滤波器设计
- Preprocess类: notch滤波、filterBank、detrend、均值计算、标准差计算
- TRCA类: TRCA算法实现
- utils: 包含数据函数、调试函数，其中一部分数据函数应该放到别的类里面

### 测试
- dllvalid.py: 使用[SSVEP-AnaTool](https://github.com/pikipity/SSVEP-Analysis-Toolbox)测试, 需确认lib版本, 也可使用本repo中提供的lib
- 使用Wearable-SSVEP(wet)数据集测试, 其中使用SSVEPAnalysisToolbox库测试时, 需要确认库get_data方法截取的数据是否正确, 需比照lib代码和数据集说明