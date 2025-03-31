clear all
clc
close
%
load('C_Gamma_30Mar.mat');
%
%% Calibration disturbance dataset for each agent
mu = [0; 0];           % Mean vector (zero mean)
Sigma = 0.001 * eye(2); % Covariance matrix (0.05 * I_2)
N = 100;
n=2;
Ncal = 1000;
theta=0.05;
GammaNN = tril(ones(N),-1);

GammannValue= -0.005*eye(2);

Gamma = kron(GammaNN,GammannValue);

A = eye(2); B = eye(2);
Abm = zeros(N * n, N * n);
% Contstruct Abm and Bbm
for k = 1:N
    Abm((k-1)*n+1:k*n, (k-1)*n+1:k*n) = eye(n);
end
for row = 2:N
    for col = 1:row-1
        Abm((row-1)*n+1:row*n, (col-1)*n+1:col*n) = A^(row-col);
    end
end
for k = 1:N
    Abm((k-1)*n+1:k*n, (k-1)*n+1:k*n) = eye(n);
end
for row = 2:N
    for col = 1:row-1
        Abm((row-1)*n+1:row*n, (col-1)*n+1:col*n) = A^(row-col);
    end
end
Bbm=Abm*kron(eye(N),B);

for i=1:10
    for j=1:Ncal
        disturbance_sequence{i,j} = mvnrnd(mu, Sigma, N)';
        error_tr_cal{i,j}=(Abm+Bbm*Gamma)*reshape(disturbance_sequence{i,j},[2*N 1]);
    end
end

%% introduce nonconformity scores
E=[];
for j=1:Ncal
    y1=c1*norm(error_tr_cal{1,j},inf); y2=c2*norm(error_tr_cal{2,j},inf); y3=c3*norm(error_tr_cal{3,j},inf);
    y4=c4*norm(error_tr_cal{4,j},inf); y5=c5*norm(error_tr_cal{5,j},inf); y6=c6*norm(error_tr_cal{6,j},inf);
    y7=c7*norm(error_tr_cal{7,j},inf); y8=c8*norm(error_tr_cal{8,j},inf); y9=c9*norm(error_tr_cal{9,j},inf);
    y10=c10*norm(error_tr_cal{10,j},inf); 
    %
    y123=c123*norm([error_tr_cal{1,j};error_tr_cal{2,j};error_tr_cal{3,j}],inf); y15=c15*norm([error_tr_cal{1,j};error_tr_cal{5,j}],inf); y34=c34*norm([error_tr_cal{3,j};error_tr_cal{4,j}],inf);
    y45=c45*norm([error_tr_cal{4,j};error_tr_cal{5,j}],inf); y56=c56*norm([error_tr_cal{5,j};error_tr_cal{6,j}],inf); y47=c47*norm([error_tr_cal{4,j};error_tr_cal{7,j}],inf);
    y68=c68*norm([error_tr_cal{6,j};error_tr_cal{8,j}],inf); y69=c69*norm([error_tr_cal{6,j};error_tr_cal{9,j}],inf); y78=c78*norm([error_tr_cal{7,j};error_tr_cal{8,j}],inf);
    y910=c910*norm([error_tr_cal{9,j};error_tr_cal{10,j}],inf); y810=c810*norm([error_tr_cal{8,j};error_tr_cal{10,j}],inf);
    %
    y = [y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y123 y15 y34 y45 y56 y47 y68 y69 y78 y910 y810];
    %
    E=[E max(y)];
end
%
Quant=quantile(E,1-theta);
%
%% STL robustness tightening
STL1 = Quant/c1;
STL2 = Quant/c2;
STL3 = Quant/c3;
STL4 = Quant/c4;
STL5 = Quant/c5;
STL6 = Quant/c6;
STL7 = Quant/c7;
STL8 = Quant/c8;
STL9 = Quant/c9;
STL10= Quant/c10;
STL123=Quant/c123;
STL15= Quant/c15;
STL34= Quant/c34;
STL45= Quant/c45;
STL56= Quant/c56;
STL47= Quant/c47;
STL68= Quant/c68;
STL69= Quant/c69;
STL78= Quant/c78;
STL910=Quant/c910;
STL810=Quant/c810;


STL_array = [STL1, STL2, STL3, STL4, STL5, STL6, STL7, STL8, STL9, STL10, ...
             STL123, STL15, STL34, STL45, STL56, STL47, STL68, STL69, STL78, STL910, STL810];

% Find the maximum value
max_STL = max(STL_array)
min_STL = min(STL_array)







