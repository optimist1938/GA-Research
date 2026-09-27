# The reference model

A line-by-line account of `matlab/SimpleNonLinearModel_ML.m`, what the Python port does
differently, and why the differences are exact rather than approximate.

## The model

```
y(k) = BL * sum_m  sum_{n_1..n_D}  c_{m,n}  *  x_1(k - s_m)  *  prod_p T_{n_p}( |x_p(k - l_m - u_{p,m})| )
```

| symbol | meaning | where it comes from |
|---|---|---|
| `k` | sample index | 221000 samples at `FsLow = 368.64 MHz` |
| `D` | dimension of the nonlinearity | `Dim`, = number of envelopes fed in |
| `x_p` | envelope of carrier `p`, unit peak | `x` rows, normalised |
| `x_1` | the carrier being modelled | first row; also `xRef` before normalisation |
| `m` | index over *parts* | 15 columns of the part model |
| `s_m` | delay of the linear carrier copy | part model row 1 |
| `l_m` | common delay of the envelope term | part model row 2 |
| `u_{p,m}` | extra delay on `|x_p|` | part model rows 3.. |
| `T_n` | Chebyshev polynomial, first kind | `gen_spl(..., Sp_ord=6)` |
| `c_{m,n}` | complex coefficients | solved by least squares |
| `BL` | band-limitation FIR, 33 taps | `fir1(32, 0.36*2, 'low')` |

Each part contributes `prod_p (n_basis[p] + 1)` coefficients. With 15 parts and Chebyshev
order 8 in two dimensions that is `15 * 9 * 9 = 1215`.

## The part model

A `(2 + D, M)` integer matrix. `PartModel.default(dim)` returns the structure shipped in
`NonLinearProblemSimple.m`, truncated to `dim`:

```
s   [ 0   2  -1   1   2   0   8   3   0   1   1  10   0  -2   2 ]
l   [ 1   2   0   1   1   0   0   1   1   2  -1   0   2   0   2 ]
u1  [-1  -2  -1   0   1   1   4   2   0  -2   0   0   0  -2  -1 ]
u2  [ 0   1   1   0   1   0   0   1   0   0   1   1   0   1   1 ]
u3  [ 0   0   1   1   0   1   0   0   1   1   0   0   1   0   1 ]   <- ours, see below
```

Rows 1–4 are exactly as supplied. **Row 5 is ours.** `GeoData_TB` is tri-band but the archive
only ships a 2-dimensional structure, so the third band's delays had to be chosen; row 5 is
row 4 shifted by one tap. Any 3-band result depends on that choice.

The delay structure matters more than the polynomial order. Sweeping the Chebyshev order on
the A+B model, in-sample:

| order per dimension | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| coefficients | 60 | 135 | 240 | 375 | 735 | 1215 |
| NMSE (dB) | -18.809 | -22.243 | -23.084 | -23.682 | -24.028 | -24.153 |

Order 2 with 135 coefficients is within 1.9 dB of order 8 with 1215. `PartModel.grid` builds
regular delay grids if you want to search the structure instead of the order.

## The Chebyshev basis

`gen_spl(2^15, 2^15/N, 1, 6)` builds an `(N+1) x 2^15` lookup table. Row `k` is `T_k`
evaluated on the magnitude grid `(0:2^15-1) + 0.5` mapped affinely to `[-1, 1)`. The model
indexes it with `round(|x| * absScale * 2^15) + 1`, with `absScale = 0.99` keeping the index
inside the table.

The knot arithmetic at the top of `gen_spl.m` only determines the row count for the
Chebyshev branch — the spline branches (`Sp_ord` 0..5) use it properly, but branch 6 ignores
the knots. `dpd.matlab.gen_spl_chebyshev` implements branch 6 only.

`MemoryPolynomialPA(quantise=True)` reproduces the lookup exactly. `quantise=False`
evaluates the Chebyshev argument continuously, which makes the model differentiable with
respect to the envelope as well as the coefficients. The two differ by **6.7e-5 dB** of NMSE
on this data; the difference signal sits at -78 dB.

## Three reorganisations, all exact

**Fold `l` into the magnitude delays.** MATLAB forms the basis product from `|x_p(k - u_p)|`
and *then* delays the product by `l`. Delay is linear and commutes with a pointwise product,
so the port uses `|x_p(k - l - u_p)|` directly and skips an array copy per basis term.

**Sum before filtering.** MATLAB convolves each of the 1215 basis terms with `BL` and then
takes a coefficient-weighted sum. Convolution is linear, so

```
sum_j c_j (BL * v_j)  ==  BL * (sum_j c_j v_j)
```

and the port convolves once instead of 1215 times. Full-record forward pass: 0.7 s.
`tests/test_matlab_parity.py::test_summed_then_filtered_equals_reference_column_form` checks
the two agree to 1e-10.

**Stream the Gram matrix.** MATLAB materialises the full regressor matrix `U` before forming
`U'U`. At `D = 3` that is 38 GB. `ChebyshevLUTFeatures.gram` accumulates `U'U` and `U'd` over
time blocks with a halo for the convolution, which is identical up to summation order.

## Column ordering

`lin2tensor.m` is a 1-based, column-major `ind2sub`: the *first* basis index varies fastest.
`MemoryPolynomialPA` stores coefficients as an `(M, n_1, .., n_D)` tensor and
`flat_coef()` / `load_flat_coef()` convert to and from MATLAB's flat ordering with a Fortran-
order reshape. Get this wrong and the NMSE is unchanged but the coefficients are permuted,
so they cannot be compared with a MATLAB solution.

## Conditioning and the solver

`U'U` has condition number ~1e21 at `D = 2` (rank 1168 of 1215) and ~4e21 at `D = 3` (rank
3269 of 3375). Forming the normal equations squares the condition number of `U`, so most of
the available precision is gone before the solve starts. The pseudo-inverse hides this by
truncating the near-null singular values: the *prediction* is stable, but the coefficient
vector is only defined up to the numerical null space.

This is why matching MATLAB's `pinv` tolerance rule matters. MATLAB uses
`tol = max(size(A)) * eps(norm(A))`, i.e. `rcond = max(m, n) * eps ~= 2.7e-13` for a
1215x1215 matrix. NumPy defaults to `rcond = 1e-15`, ~270x smaller, which keeps 15 more
near-null directions and yields a minimum-norm solution 13x larger. Same prediction,
different coefficients. `dpd.solvers.matlab_pinv_solve` implements MATLAB's rule.

At `D = 3` the difference shows up in generalisation:

| solver | train | val | coefficient norm |
|---|---|---|---|
| `pinv`, rcond 1e-15 | -24.988 dB | -23.556 dB | 1.1e3 |
| `pinv`, MATLAB rule | -24.981 dB | -23.847 dB | 2.0e1 |
| `ridge`, alpha 1e-6 | -24.906 dB | **-24.473 dB** | 7.9e0 |

The pseudo-inverse buys 0.08 dB in-sample and gives back 0.6 dB out of sample. Prefer
`--solver ridge` at `D = 3`; held-out NMSE is flat to ~0.1 dB over `alpha` in 1e-7..1e-4, so
the default of 1e-6 is not delicate.

If you need a better-conditioned identification, the right move is to avoid the normal
equations entirely and run a QR or SVD least squares on `U` directly — `cond(U)` is the square
root of `cond(U'U)`, around 6e10. `ChebyshevLUTFeatures.matrix()` will materialise `U` for
small configurations. This is not currently used by any script.

## Verification against the original code

`matlab/run_geodata.m` runs the untouched `.m` files on `GeoData_TB` under Octave:

```bash
apt-get install -y octave octave-signal
cd matlab && BAND=A BANDS=AB NB="[8 8]" octave --no-gui --quiet run_geodata.m
```

| | NMSE | coefficients |
|---|---|---|
| Octave, original `.m` files | -24.15657 dB | — |
| Python port | -24.15318 dB | agree to 0.5% |

The 0.003 dB gap and the 0.5% coefficient difference both come from the rank deficiency:
the two SVD implementations retain slightly different near-null directions.

## Helpers the archive does not ship

`delay.m`, `nmse.m` and `progress.m` live on the original author's MATLAB path. The versions
in `matlab/` are ours, and the port depends on them:

- **`delay(x, n)`** is implemented as a **circular** shift, `circshift(x, [0 n])`. The
  alternative (zero-padded) differs on at most 10 of 221000 samples, which is below 1e-4 dB
  of NMSE. This is the least certain of the three assumptions, and the one most worth
  confirming against the original.
- **`nmse(ref, err)`** is `10*log10(sum|err|^2 / sum|ref|^2)`. Corroborated by `eRefA`
  scoring -33.45 dB under it, a plausible figure for a shipped PA model. See
  [metrics.md](metrics.md).
- **`progress`** is a progress bar; ours is a no-op.
