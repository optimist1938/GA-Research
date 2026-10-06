function plot_psd4(x, y, z, f, N, Fs, nFig, hold_on)
% Task: drawing of PSD plots (0..Fs/2 - for real data, -Fs/2..Fs/2 - for complex data)
% Parameters:
%       x, y, z, f: vectors with signals
%			(x - blue, y - red, z - green, f - black);
%		N: number of points in spectrum (default 2048);
%		Fs: sampling frequency
% Return:
% trulala

if(nargin < 1)
    error('function should have at least 1 argument');
end
if(nargin < 2)
    y = [];
end
if(nargin < 3)
    z = [];
end
if(nargin < 4)
    f = [];
end
if(nargin < 5)
    N = 2048;
end
if(nargin < 6)
    Fs = 245.76e6*3/2;
end
if(nargin < 7)
    nFig = -1;
end
if(nargin < 8)
    hold_on = 0;
end

NH = floor(N / 2);

%% calculation of spectrums
% 1st spectrum
xPsd = psd(x, N, Fs).';
if(isreal(x))
   xLog = 10 * log10(xPsd);
   xFreq = (0 : NH) / N * Fs * 1e-6;
else
   xPsd = [xPsd(NH + 1 : N) xPsd(1 : NH)];
   xLog = 10 * log10(xPsd);
   xFreq = ((0 : N - 1) - NH) / N * Fs * 1e-6;
end
% 2nd spectrum
if(~isempty(y))
    yPsd = psd(y, N, Fs).';
    if(isreal(y))
        yLog = 10 * log10(yPsd);
        yFreq = (0 : NH) / N * Fs * 1e-6;
    else
        yPsd = [yPsd(NH + 1 : N) yPsd(1 : NH)];
        yLog = 10 * log10(yPsd);
        yFreq = ((0 : N - 1) - NH) / N * Fs * 1e-6;
    end
end
% 3rd spectrum
if(~isempty(z))
    zPsd = psd(z, N, Fs).';
    if(isreal(z))
        zLog = 10 * log10(zPsd);
        zFreq = (0 : NH) / N * Fs * 1e-6;
    else
        zPsd = [zPsd(NH + 1 : N) zPsd(1 : NH)];
        zLog = 10 * log10(zPsd);
        zFreq = ((0 : N - 1) - NH) / N * Fs * 1e-6;
    end
end
% 4rd spectrum
if(~isempty(f))
    fPsd = psd(f, N, Fs).';
    if(isreal(f))
        fLog = 10 * log10(fPsd);
        fFreq = (0 : NH) / N * Fs * 1e-6;
    else
        fPsd = [fPsd(NH + 1 : N) fPsd(1 : NH)];
        fLog = 10 * log10(fPsd);
        fFreq = ((0 : N - 1) - NH) / N * Fs * 1e-6;
    end
end
%% drawing of plots
% 1st plot
% 1st plot
if nFig>0
    if hold_on == 0
        clf(nFig);
    else
        hold on;
    end
    figure(nFig);
else
    figure();
end
plot(xFreq, xLog, 'b-');
hold on; grid on;
% 2nd plot
if(~isempty(y))
    plot(yFreq, yLog, 'r-');
end
% 3rd plot
if(~isempty(z))
    plot(zFreq, zLog, 'g-');
end
% 4rd plot
if(~isempty(f))
    plot(fFreq, fLog, 'k-');
end

xlabel('frequency, MHz');
ylabel('power spectrum, dB');
