function ok = dpd_selfcheck(model_file)
%DPD_SELFCHECK  Check that dpd_gated_tcn reproduces the Python model on the bundled vector.
%
%   OK = DPD_SELFCHECK('dpd_tcn16x4.mat') runs the model on model.x_check (4096 samples of the
%   test part of GeoData_TB) and compares with model.d_check, computed by the PyTorch original.
%   The training ran in single precision, so a relative error around 1e-6 is expected.
%   It also checks that chunked processing gives the same result.

if nargin < 1, model_file = 'dpd_tcn16x4.mat'; end
S = load(model_file);
m = S.model;
D = dpd_gated_tcn(m.x_check, m);
err = max(abs(D(:) - m.d_check(:))) / max(abs(m.d_check(:)));
Dc = dpd_gated_tcn(m.x_check, m, 1000);
err_chunk = max(abs(Dc(:) - D(:))) / max(abs(D(:)));
ok = err < 1e-4 && err_chunk < 1e-12;
fprintf('%s: %d parameters, %d MAC per sample\n', model_file, round(m.n_params), round(m.macs_per_sample));
fprintf('  max relative error vs Python: %.2e\n', err);
fprintf('  chunked vs whole:             %.2e\n', err_chunk);
if ok
    fprintf('  OK\n');
else
    fprintf('  MISMATCH\n');
end
end
