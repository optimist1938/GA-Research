function D = dpd_gated_tcn(X, model, chunk)
%DPD_GATED_TCN  Tri-band DPD correction: phase-equivariant gated TCN.
%
%   D = DPD_GATED_TCN(X, MODEL) returns the predistortion correction D (3 x N, complex)
%   for the baseband envelopes X (3 x N, complex). Rows are bands A, B, C; units are those
%   of xA, xB, xC in GeoData_TB.mat, and D is in the units of dA, dB, dC.
%   MODEL is the struct saved in dpd_tcn*.mat:  S = load('dpd_tcn16x4.mat'); model = S.model;
%
%   D = DPD_GATED_TCN(X, MODEL, CHUNK) processes CHUNK samples at a time (default 65536)
%   with an overlap, so long records fit in memory. The result does not depend on CHUNK.
%
%   The model, for each band c:
%       D(c,k) = sum_{m=-M..M} x_c(k-m) * g_{c,m}(k)
%   The complex gains g come from a small dilated 1-D convolutional network that sees only
%   phase-free quantities of all three bands: |x_b(k)|^2 and x_b(k)*conj(x_b(k-j)), j = 1..J.
%   Rotating the phase of any band b by a constant therefore rotates D(b,:) by the same phase
%   and leaves the other bands unchanged.
%
%   The model is non-causal: D(:,k) uses inputs up to model.receptive + model.memory samples
%   ahead. Plain MATLAB (R2016b+) or GNU Octave, no toolboxes.

if nargin < 3 || isempty(chunk), chunk = 65536; end
if size(X, 1) ~= 3, error('dpd_gated_tcn:input', 'X must be 3 x N (rows: bands A, B, C)'); end
N = size(X, 2);
% Samples on each side of a chunk that can influence its outputs (+2 spare).
halo = model.receptive + model.memory + model.max_lag_diff + 2;
D = complex(zeros(3, N));
for s = 1:chunk:N
    e = min(s + chunk - 1, N);
    lo = max(1, s - halo);
    hi = min(N, e + halo);
    Dc = forward(X(:, lo:hi), model);
    D(:, s:e) = Dc(:, (s - lo + 1):(e - lo + 1));
end
end


function D = forward(X, m)
n = size(X, 2);
X = double(X) .* m.scale;                       % per-band scale fixed at training
F = local_invariants(X, m.max_lag_diff);        % (3 + 6J) x n, phase-free
F = (F - m.mu) ./ m.sd;
h = m.W_in * F + m.b_in;                        % 1x1 conv: C x n
for i = 1:m.levels                              % residual blocks, dilation 1, 2, 4, ...
    a = gelu(h);
    a = conv_dil3(a, m.W1(:, :, :, i), m.b1(:, i), 2^(i - 1));
    a = gelu(a);
    a = m.W2(:, :, i) * a + m.b2(:, i);
    h = h + a;
end
g = m.W_out * h + m.b_out;                      % 3 bands x T taps x (re, im), per sample
T = 2 * m.memory + 1;
G = reshape(g, [2, T, 3, n]);
gain = reshape(complex(G(1, :, :, :), G(2, :, :, :)), [T, 3, n]);
D = complex(zeros(3, n));
for t = 1:T
    shift = t - 1 - m.memory;                   % carrier tap x_c(k - shift), shift = -M..M
    D = D + zdelay(X, shift) .* reshape(gain(t, :, :), [3, n]);
end
D = D ./ m.scale;
end


function F = local_invariants(X, J)
% Rows: |x|^2 (A,B,C); then for j = 1..J: Re(x conj(x(k-j))) (A,B,C), Im(...) (A,B,C).
F = abs(X).^2;
for j = 1:J
    p = X .* conj(zdelay(X, j));
    F = [F; real(p); imag(p)]; %#ok<AGROW>
end
end


function y = conv_dil3(a, W, b, dil)
% Kernel-3 dilated cross-correlation with zero padding ('same' length), as in PyTorch
% Conv1d(kernel_size=3, dilation=dil, padding=dil): tap j reads a(:, k + (j-2)*dil).
n = size(a, 2);
y = repmat(b, 1, n);
for j = 1:3
    y = y + W(:, :, j) * zdelay(a, -(j - 2) * dil);
end
end


function y = zdelay(x, k)
% Delay along columns with zeros: y(:, t) = x(:, t - k); k < 0 looks ahead.
n = size(x, 2);
y = zeros(size(x));                 % becomes complex on assignment if x is complex
if abs(k) >= n, return; end
if k >= 0
    y(:, k+1:n) = x(:, 1:n-k);
else
    y(:, 1:n+k) = x(:, 1-k:n);
end
end


function y = gelu(x)
y = 0.5 * x .* (1 + erf(x / sqrt(2)));
end
