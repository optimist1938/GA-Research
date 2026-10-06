function v = nmse(ref, err)
v = 10*log10(sum(abs(err(:)).^2) / sum(abs(ref(:)).^2));
end
