function [Y] = spln1(X, Knt)

% 1-order spline calculation
% ----------------------------------------INPUT
% X       - Input signal
% Knt     - Splines knots
% ---------------------------------------OUTPUT
% Y       - Spline
% ------------------------------------------END
% Initialization
Y = 0*X;

% Index calculation
Ind1 = find((X >= Knt(1))&(X < Knt(2)));
Ind2 = find((X >= Knt(2))&(X < Knt(3)));
Int = Knt(2)-Knt(1);

% Calculation
Y(Ind1) = ((X(Ind1)-Knt(1))/Int);
Y(Ind2) = -(X(Ind2)-Knt(2))/Int+1;
