% 读取 CSV 文件
data = csvread('filter_a_b_dll.csv');

% 获取滤波器系数的行数
[num_rows, ~] = size(data);

% 采样频率
fs = 250;

% 循环读取每组滤波器系数并绘制响应曲线
for i = 1:2:num_rows-1
    % 获取滤波器系数
    A = data(i, :);
    B = data(i+1, :);
    
    % 计算频率响应
    [H, w] = freqz(B, A);
    
    % 计算幅频响应和相频响应
    Hf = abs(H);  % 取幅度值实部
    Hx = angle(H);  % 取相位值对应相位角
    
    % 绘制幅频响应曲线
    figure;
    subplot(2, 1, 1);
    plot(w * fs / (2 * pi), 20 * log10(Hf));  % 幅值变换为分贝单位
    title(['滤波器组 ' num2str((i+1)/2) ' 幅频特性曲线']);
    xlabel('频率 (Hz)');
    ylabel('幅值 (dB)');
    grid on;
    
    % 绘制相频响应曲线
    subplot(2, 1, 2);
    plot(w * fs / (2 * pi), Hx);
    title(['滤波器组 ' num2str((i+1)/2) ' 相频特性曲线']);
    xlabel('频率 (Hz)');
    ylabel('相位 (弧度)');
    grid on;
end