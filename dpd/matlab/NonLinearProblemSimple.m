
% input data Sample Rate
Fs = 122.88*4;
% Load signal
load('dov2.mat');

slotSize = size(PDin,2);
sml = 1:slotSize-1;
UpSample = 2;
if UpSample == 1
    HBF = [0 1 0];
else
    HBF = fir1(4096,1/UpSample,'low');
end
PDinA = round(conv(upsample(PDin(1,sml),UpSample)*UpSample, HBF, 'same'));
PDerrA = round(conv(upsample(PDout(1,sml),UpSample)*UpSample, HBF, 'same'));
PDdpdA = round(conv(upsample(PDdpd(1,sml),UpSample)*UpSample, HBF, 'same'));
PDerrA = PDdpdA-PDerrA;

%% Special filter for nonlinearity bandwidth limitation
flen = 32;
BL =  fir1(flen,0.36*2,'low');

slotSize = size(PDinA,2);

%% Residual error from data file as reference model error (just for comparisson)
eRefA = round(conv(PDout(1,sml) - PDin(1,sml), BL,'same'));

%% Prepare input signal x and desired signal
x = zeros(UpSample, slotSize/UpSample);
for pp = 1:UpSample
   x(pp,:) = PDinA(pp:UpSample:end); 
end
dA = round(conv(PDerrA(1:UpSample:end), BL, 'same'));

Fs = Fs*1e6;

g = 2^(-15); %1/std(PDin);
x = x * g;
dA = dA * g;

xRef = x(1,:);

% Scale factor for Chebyshev basis Tensor
for pp = 1:UpSample
    k = (1/max(abs(x(pp,:))))*1;
    x(pp,:) = x(pp,:).*k;
end

%% Model configuration as delays and basis function numbers
% Currrent realsization based on 2 Dimensional for non-linear function
ModelBasisFuncNum = [8 8];
Dim = length(ModelBasisFuncNum);

PartModel = [0     2    -1     1     2     0     8     3     0     1     1    10     0    -2     2;
             1     2     0     1     1     0     0     1     1     2    -1     0     2     0     2;
            -1    -2    -1     0     1     1     4     2     0    -2     0     0     0    -2    -1;
             0     1     1     0     1     0     0     1     0     0     1     1     0     1     1];

figure(15);
%% Start Optimization Problem 
[newErr, coefs, ~] = SimpleNonLinearModel_ML(Dim, xRef, x, dA, PartModel, ModelBasisFuncNum, 0.99, BL, 0);          

plot_psd4(xRef/g,round(dA/g),round(newErr/g),eRefA,2048,Fs,15);drawnow;
fprintf('\n Dimension of NonLinearity is: %d\n',Dim);
fprintf('   Number of model coefficients is: %d\n',length(coefs));
