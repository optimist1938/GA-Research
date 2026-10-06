function [eLSopt, cLSopt, yLSopt] = SimpleNonLinearModel_ML(Dim, xRef, xA, dA, PartModel, splNum1D, absScale, BL, tol)

BandSize = 2^15 ./(splNum1D);

%% Generate matrix with basis Chebyshev function (can be chenged if it nessesary)
%  gen_spl(2^15,BandSize(pp),1,1) - First order Splines
%  gen_spl(2^15,BandSize(pp),1,2) - Second order Splines
%  gen_spl(2^15,BandSize(pp),1,6) - Chebyshev function - is optimimal basis (maximum approximations with minimum coefficients

for pp = 1: length(splNum1D)
     splMtrx.mtrx(pp).spls = gen_spl(2^15,BandSize(pp),1,6);
end
splNum1D = (splNum1D + 1)*1;
NumberSplines = prod(splNum1D);

%% matrix Vand initialization
    U = zeros(length(xA(1,:)), NumberSplines * size(PartModel,2));
    Uidx = 0;
    progress;
    % Cycle for each non-linear components
    for dd = 1 : size(PartModel,2)
        absX = zeros(Dim, length(xA(1,:)));
        progress(1, (dd/size(PartModel,2))*100);
        
        % Delay for model X(k-sd) * F(|X(k-p-ld)|)
        sd = PartModel(1,dd);
        ld = PartModel(2,dd);

        for pp = 1: size(xA,1)
            absX(pp,:) = round(abs(delay(xA(pp,:),PartModel(3 + (pp-1),dd))) * absScale*2^15) + 1;
        end             
       pidx = 0;
       while pidx < prod(splNum1D)
            pidx = pidx + 1; 
            CurrentOrd = (lin2tensor(splNum1D, pidx));
            splVector3D = 1;
            for pp = 1: length(splNum1D)
                splVector3D = splVector3D .* splMtrx.mtrx(pp).spls(CurrentOrd(pp), absX(pp,:));
            end
            Uidx = Uidx + 1;
            U(:, Uidx) = conv(delay(splVector3D, ld).*delay(xA(1,:), sd), BL, 'same').';
       end

    end
    fprintf('\n');
    disp('Finished LUT Vand calculation');
    RX = U' * U;
    RY = U' * dA.';
    fprintf('Finished RX,RY calculation. Start Solution...');
    tic
    %% Least Squares solution
    if tol == 0
        cLSopt = pinv(RX) * RY;
    else
        cLSopt = pinv(RX,tol) * RY;
    end
   %% Calculate model output 
   yLSopt = U*cLSopt;
   %% Calculate residual error from model
   eLSopt = dA - yLSopt.';
   fprintf('%.f sec\n',toc);        
   clear RX RY U;
   fprintf('NMSE: %.5f \n',nmse(xRef,eLSopt));
end  
