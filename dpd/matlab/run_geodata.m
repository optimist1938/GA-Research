pkg load signal;
load('/tmp/claude-0/-home-user-GA-Research/3bf0684e-472c-5194-9667-1ec0c3d09880/scratchpad/unz/geodata/GeoData_TB.mat');
BL = fir1(32, 0.36*2, 'low');
band = getenv('BAND'); if isempty(band), band = 'A'; end
bands = getenv('BANDS'); if isempty(bands), bands = 'AB'; end
nb = str2num(getenv('NB'));
eval(sprintf('d = d%s; eR = eRef%s;', band, band));
X = [];
for i = 1:length(bands)
  eval(sprintf('X = [X; x%s];', bands(i)));
end
xRef = X(1,:);
for pp = 1:size(X,1)
  X(pp,:) = X(pp,:) * (1/max(abs(X(pp,:))));
end
Dim = length(nb);
PartModel = [0     2    -1     1     2     0     8     3     0     1     1    10     0    -2     2;
             1     2     0     1     1     0     0     1     1     2    -1     0     2     0     2;
            -1    -2    -1     0     1     1     4     2     0    -2     0     0     0    -2    -1;
             0     1     1     0     1     0     0     1     0     0     1     1     0     1     1;
             0     0     1     1     0     1     0     0     1     1     0     0     1     0     1];
PartModel = PartModel(1:2+Dim, :);
tic
[eLS, cLS, yLS] = SimpleNonLinearModel_ML(Dim, xRef, X, d, PartModel, nb, 0.99, BL, 0);
fprintf('OCTAVE band=%s bands=%s nb=%s ncoef=%d\n', band, bands, mat2str(nb), length(cLS));
fprintf('OCTAVE NMSE(model)   = %.5f dB\n', nmse(xRef, eLS));
fprintf('OCTAVE NMSE(nomodel) = %.5f dB\n', nmse(xRef, d));
fprintf('OCTAVE NMSE(eRef)    = %.5f dB\n', nmse(xRef, eR));
save('-v7', sprintf('oct_out_%s_%s.mat', band, bands), 'cLS', 'yLS');
toc
