function res = run_example_geodata(geodata_file, model_file, show_psd)
%RUN_EXAMPLE_GEODATA  Apply the model to GeoData_TB.mat and report NMSE per band.
%
%   RES = RUN_EXAMPLE_GEODATA('GeoData_TB.mat', 'dpd_tcn16x4.mat') computes the correction
%   D for the whole record and prints, on the last 20% of the samples (the part not used for
%   training), the NMSE of the residual d - D per band:
%       vs x : 10*log10(sum|d - D|^2 / sum|x|^2), as nmse(xRef, e) in SimpleNonLinearModel_ML
%       vs d : 10*log10(sum|d - D|^2 / sum|d|^2)
%   and the same "vs x" figure for eRef. RUN_EXAMPLE_GEODATA(..., true) also plots the PSD of
%   x, d, d - D and eRef per band (Welch, Hann window, plain MATLAB).
%
%   For another record of the same PA, pass a file with the same variables (xA, xB, xC, dA,
%   dB, dC, optionally eRefA/B/C and FsLow) and the same units.

if nargin < 1 || isempty(geodata_file), geodata_file = 'GeoData_TB.mat'; end
if nargin < 2 || isempty(model_file), model_file = 'dpd_tcn16x4.mat'; end
if nargin < 3, show_psd = false; end

G = load(geodata_file);
S = load(model_file);
X = [G.xA(:).'; G.xB(:).'; G.xC(:).'];
Dref = [G.dA(:).'; G.dB(:).'; G.dC(:).'];
has_eref = isfield(G, 'eRefA');
if has_eref, E = [G.eRefA(:).'; G.eRefB(:).'; G.eRefC(:).']; end
fs = 1; if isfield(G, 'FsLow'), fs = double(G.FsLow); end

tic;
D = dpd_gated_tcn(X, S.model);
t = toc;
N = size(X, 2);
test = (floor(0.8 * N) + 1):N;
bands = 'ABC';
fprintf('%s on %s: %d samples in %.1f s, test = last %d samples\n', model_file, geodata_file, N, t, numel(test));
fprintf('band   NMSE vs x   NMSE vs d   eRef vs x\n');
res = struct('band', {}, 'nmse_vs_x', {}, 'nmse_vs_d', {}, 'eref_vs_x', {});
for c = 1:3
    e = Dref(c, test) - D(c, test);
    r.band = bands(c);
    r.nmse_vs_x = nmse_db(X(c, test), e);
    r.nmse_vs_d = nmse_db(Dref(c, test), e);
    r.eref_vs_x = NaN;
    if has_eref, r.eref_vs_x = nmse_db(X(c, test), E(c, test)); end
    res(c) = r;
    fprintf('  %s    %8.2f    %8.2f    %8.2f\n', r.band, r.nmse_vs_x, r.nmse_vs_d, r.eref_vs_x);
end

if show_psd
    figure;
    for c = 1:3
        subplot(3, 1, c); hold on; grid on;
        [f, p] = welch_psd(X(c, test), fs); h = plot(f / 1e6, p, 'b');
        [f, p] = welch_psd(Dref(c, test), fs); h(end + 1) = plot(f / 1e6, p, 'r');
        [f, p] = welch_psd(Dref(c, test) - D(c, test), fs); h(end + 1) = plot(f / 1e6, p, 'g');
        legend_items = {'x', 'd', 'd - D'};
        if has_eref
            [f, p] = welch_psd(E(c, test), fs); h(end + 1) = plot(f / 1e6, p, 'k');
            legend_items{end + 1} = 'eRef';
        end
        title(sprintf('band %s', bands(c))); xlabel('MHz'); ylabel('dB/Hz');
        legend(h, legend_items, 'Location', 'eastoutside');
    end
end
end


function v = nmse_db(ref, err)
v = 10 * log10(sum(abs(err(:)).^2) / sum(abs(ref(:)).^2));
end


function [f, p] = welch_psd(x, fs)
nfft = 2048;
w = 0.5 - 0.5 * cos(2 * pi * (0:nfft-1)' / nfft);      % Hann, periodic
x = x(:);
starts = 1:(nfft / 2):(numel(x) - nfft + 1);
acc = zeros(nfft, 1);
for s = starts
    acc = acc + abs(fft(x(s:s + nfft - 1) .* w)).^2;
end
p = fftshift(acc / numel(starts) / (sum(w.^2) * fs));
p = 10 * log10(p + realmin);
f = ((0:nfft-1)' - nfft / 2) * fs / nfft;
end
