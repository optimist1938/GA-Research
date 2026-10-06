function y = delay(x, n)
% circular sample delay: y(k) = x(k-n)
y = circshift(x, [0 n]);
end
