function [Y] = spln2(X, Knt)

% 0-order spline calculation
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
Ind3 = find((X >= Knt(3))&(X < Knt(4)));
Int = Knt(2)-Knt(1);

% Calculation
Y(Ind1) = ((X(Ind1)-Knt(1))/Int).^2;
Y(Ind2) = -2*((X(Ind2)-0.5*(Knt(2)+Knt(3)))/Int).^2+1.5;
Y(Ind3) = ((X(Ind3)-Knt(4))/Int).^2;

Y = Y/1.5;