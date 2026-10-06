function [v] = lin2tensor(T,ndx)
% Calculate Multiple (Tensor) indexes from linear index.
%

nout = length(T);
T = double(T);


k = cumprod(T);
for i = nout:-1:1
    if i == 1
        elem = T(1);
    else
        elem = k(i - 1); 
    end
    vi = rem(ndx-1, elem) + 1;
    vj = (ndx - vi)/elem + 1;
    if i == 1
        v(i) = double(vi);
    else
        v(i) = double(vj);
    end
    ndx = vi;
end



