function [Smag] = gen_spl(Mg_max, Mg_int, Mg_bit, Sp_ord)

% Splines array generation
% ----------------------------------------INPUT
% Mg_max  - Magnitude maximum value
% Mg_int  - Single magnitude interval
% Mg_bit  - 1-bit magnitude index 
% ---------------------------------------OUTPUT
% Smag    - Splines array
% ------------------------------------------END

% Initialisation
Max_ind = fix(Mg_max/Mg_bit);
Step = fix(Mg_int/Mg_bit);

switch Sp_ord
    case 0
        Knt = 0:Step:Max_ind;
        N = length(Knt)-1;
    case 1
        Knt = -Step:Step:(Max_ind+Step);
        N = length(Knt)-2;
    case 2
        Knt = -2*Step:Step:(Max_ind+Step);
        N = length(Knt)-3;
    case 3
        Knt = -2*Step:Step:(Max_ind+Step);
        N = length(Knt)-3;
    case 4
        Knt = -3*Step:Step:(Max_ind+Step);
        N = length(Knt)-4;
    case 5
        Knt = -2*Step:Step:(Max_ind+Step);
        N = length(Knt)-3;
    case 6
        Knt = -2*Step:Step:(Max_ind+Step);
        N = length(Knt)-3;
        
end

% Splines generation
Smag = zeros(N,Max_ind);
Mg_ind = (0:Max_ind-1)+0.5;
M = N;
for k = 1:N
    switch Sp_ord
    case 0
        Smag(k,:) = spln0(Mg_ind,Knt(k:k+1));
    case 1
        Smag(k,:) = spln1(Mg_ind,Knt(k:k+2));
    case 2
        Smag(k,:) = spln2(Mg_ind,Knt(k:k+3));
    case 3
        xx = berPoly(Mg_ind/Mg_max,N-1,k-1);
        xx = xx./max(xx);
        Smag(k,:) = xx;
    case 4
        Smag(k,:) = spln3(Mg_ind,Knt(k:k+4));
    case 5
        if mod(k,2) == 0
            xx = sin((k-1)*(-pi:1/((Mg_max-1)/2/pi):pi))+1;
        else
            xx = cos((k-1)*(-pi:1/((Mg_max-1)/2/pi):pi))+1;
        end
        Smag(k,:) = xx/max(xx);
    case 6
        xx = ChebyshevFunc(2*(Mg_ind/Mg_max-0.5),k-1,1);
        Smag(k,:) = xx;
            
    end
end
% if Sp_ord ==3
%     Smag([1 N],:) = [];
% end
