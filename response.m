[H,w]=freqz(B,A);
Hf=abs(H);  %取幅度值实部
Hx=angle(H);  %取相位值对应相位角
clf
figure(1);fs=250;
plot(w*fs/(2*pi),20*log10(Hf))  %幅值变换为分贝单位
title('FilterBank-1滤波器幅频特性曲线')
figure(2)
plot(w*fs/(2*pi),Hx)
title('FilterBank-1滤波器相频特性曲线');xlabel('Hz');ylabel('dB');