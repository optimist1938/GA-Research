function [Ch] = ChebyshevFunc(x,n, Order)

if n < 2
    switch (n)
        case 0
            Ch = ones(1, length(x));
        case 1
            if (Order == 2)
                Ch = 2.*x;
            elseif (Order == 1)
                Ch = x;
            else
                error('incorrect Order');
            end
    end
else
    Ch = 2.*x.*ChebyshevFunc(x,n-1,Order) - ChebyshevFunc(x,n-2,Order);
end



end