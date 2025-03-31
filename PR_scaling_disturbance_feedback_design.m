% PR-scaling and disturbance feedback design
clear all
clc
close all
%
%% system matrices and parameters
N = 100; n=2; Ntrain=1; Ncal=1000;
%
theta_hat=min([0.9999 (1+1/Ntrain)*0.95]);
q=Ntrain*(1-theta_hat);
%
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
%% Generate Gaussian disturbance sequence training dataset for each agent
mu = [0; 0];           % Mean vector (zero mean)
Sigma = 0.05 * eye(2); % Covariance matrix (0.05 * I_2)
for i=1:10
    for j=1:Ntrain
        disturbance_sequence{i,j} = mvnrnd(mu, Sigma, N)';
    end
end
% 
%% optimization
% variables
% feedback gains
tmax=10;
GammaNN=sdpvar(N,N,'full');
GammaNn=sdpvar(N*n,N*n,'full');
Gammann=sdpvar(n,n,'full');
% error trajectories
for i=1:10
    for j=1:Ntrain
        error_tr{i,j}=sdpvar(n*N,1,'full');
    end
end
% PR-scaling
C1=sdpvar; C2=sdpvar; C3=sdpvar; C4=sdpvar; C5=sdpvar;
C6=sdpvar; C7=sdpvar; C8=sdpvar; C9=sdpvar; C10=sdpvar;
C123=sdpvar; C15=sdpvar; C34=sdpvar; C45=sdpvar;
C56=sdpvar; C47=sdpvar; C68=sdpvar; C69=sdpvar;
C78=sdpvar; C910=sdpvar; C810=sdpvar;
% Var, Y
eta=sdpvar; Y=sdpvar(Ntrain,1,'full'); z=sdpvar(Ntrain,1,'full');
% constraints
FGamma = [GammaNN==tril(ones(N),-1), GammaNn==kron(GammaNN,Gammann)];
Ferror=[];
for i=1:10
    for j=1:Ntrain
        Ferror=[Ferror, error_tr{i,j}==(Abm+Bbm*GammaNn)*reshape(disturbance_sequence{i,j},[n*N 1])];
    end
end
%
%% Run Algorithm 1 for tmax iterations
for t=1:1
    F=[];
    if t==1
        % solve for Gamma gains
        c1=1/10; c2=1/10; c3=1/10; c4=1/10; c5=1/10; c6=1/10; c7=1/10;
        c8=1/10; c9=1/10; c10=1/10; c123=0; c15=0; c34=0; c45=0; c56=0;
        c47=0; c68=0; c69=0; c78=0; c910=0; c810=0;
    end
    FG=[];
    for j=1:Ntrain
        FG=[FG, Y(j)>=c1*norm(error_tr{1,j},inf), Y(j)>=c2*norm(error_tr{2,j},inf),Y(j)>=c3*norm(error_tr{3,j},inf),...
            Y(j)>=c4*norm(error_tr{4,j},inf),Y(j)>=c5*norm(error_tr{5,j},inf),Y(j)>=c6*norm(error_tr{6,j},inf),...
            Y(j)>=c7*norm(error_tr{7,j},inf),Y(j)>=c8*norm(error_tr{8,j},inf),Y(j)>=c9*norm(error_tr{9,j},inf),...
            Y(j)>=c10*norm(error_tr{10,j},inf),Y(j)>=c123*norm([error_tr{1,j};error_tr{2,j};error_tr{3,j}],inf),...
            Y(j)>=c15*norm([error_tr{1,j};error_tr{5,j}],inf),Y(j)>=c34*norm([error_tr{3,j};error_tr{4,j}],inf),...
            Y(j)>=c45*norm([error_tr{4,j};error_tr{5,j}],inf),Y(j)>=c56*norm([error_tr{5,j};error_tr{6,j}],inf),...
            Y(j)>=c47*norm([error_tr{4,j};error_tr{7,j}],inf),Y(j)>=c68*norm([error_tr{6,j};error_tr{8,j}],inf),...
            Y(j)>=c69*norm([error_tr{6,j};error_tr{9,j}],inf),Y(j)>=c78*norm([error_tr{7,j};error_tr{8,j}],inf),...
            Y(j)>=c910*norm([error_tr{9,j};error_tr{10,j}],inf),Y(j)>=c810*norm([error_tr{8,j};error_tr{10,j}],inf),...
            z(j)>=Y(j)-eta, z(j)>=0];
    end
    F = [FG, FGamma, Ferror, eta>=0];
    options = sdpsettings('solver', 'gurobi', 'verbose', 1, 'gurobi.IterationLimit', 100);
%     options = sdpsettings('solver', 'mosek', 'verbose', 1);
    obj=eta+(1/q)*ones(1,Ntrain)*z;%  max(Y(t)-eta(t),0);
    result=optimize(F,obj,options)
    %
    GammaValue = value(GammaNn);
    for i=1:10
        for j=1:Ntrain
            error_trValue{i,j} = value(error_tr{i,j});
        end
    end
    % solve for C weights
    F=[];
    FC = [];
    FC=[FC,C1>=0,C1<=1,C2>=0,C2<=1,C3>=0,C3<=1,C4>=0,C4<=1,C5>=0,C5<=1,C6>=0,C6<=1,...
        C7>=0,C7<=1,C8>=0,C8<=1,C9>=0,C9<=1,C10>=0,C10<=1,C123>=0,C123<=1,C15>=0,C15<=1,...
        C34>=0,C34<=1,C45>=0,C45<=1,C56>=0,C56<=1,C47>=0,C47<=1,C68>=0,C68<=1,C69>=0,C69<=1,...
        C78>=0,C78<=1,C910>=0,C910<=1,C810>=0,C810<=1];
    FC=[FC, C1+C2+C3+C4+C5+C6+C7+C8+C9+C10+C123+C15+C34+C45+C56+C47+C68+C69+C78+C910+C810==1];
    for j=1:Ntrain
        FC=[FC, Y(j)>=C1*norm(error_trValue{1,j},inf), Y(j)>=C2*norm(error_trValue{2,j},inf),Y(j)>=C3*norm(error_trValue{3,j},inf),...
            Y(j)>=C4*norm(error_trValue{4,j},inf),Y(j)>=C5*norm(error_trValue{5,j},inf),Y(j)>=C6*norm(error_trValue{6,j},inf),...
            Y(j)>=C7*norm(error_trValue{7,j},inf),Y(j)>=C8*norm(error_trValue{8,j},inf),Y(j)>=C9*norm(error_trValue{9,j},inf),...
            Y(j)>=C10*norm(error_trValue{10,j},inf),Y(j)>=C123*norm([error_trValue{1,j};error_trValue{2,j};error_trValue{3,j}],inf),...
            Y(j)>=C15*norm([error_trValue{1,j};error_trValue{5,j}],inf),Y(j)>=C34*norm([error_trValue{3,j};error_trValue{4,j}],inf),...
            Y(j)>=C45*norm([error_trValue{4,j};error_trValue{5,j}],inf),Y(j)>=C56*norm([error_trValue{5,j};error_trValue{6,j}],inf),...
            Y(j)>=C47*norm([error_trValue{4,j};error_trValue{7,j}],inf),Y(j)>=C68*norm([error_trValue{6,j};error_trValue{8,j}],inf),...
            Y(j)>=C69*norm([error_trValue{6,j};error_trValue{9,j}],inf),Y(j)>=C78*norm([error_trValue{7,j};error_trValue{8,j}],inf),...
            Y(j)>=C910*norm([error_trValue{9,j};error_trValue{10,j}],inf),Y(j)>=C810*norm([error_trValue{8,j};error_trValue{10,j}],inf),...
            z(j)>=Y(j)-eta, z(j)>=0];
    end
    F = [FC, eta>=0];
    options = sdpsettings('solver', 'gurobi', 'verbose', 1);
    obj=eta+(1/q)*ones(1,Ntrain)*z;%  max(Y(t)-eta(t),0);
    result=optimize(F,obj,options)
    %
    c1=value(C1); c2=value(C2); c3=value(C3); c4=value(C4); c5=value(C5); c6=value(C6); c7=value(C7); c8=value(C8); c9=value(C9); c10=value(C10);
    c123=value(C123); c15=value(C15); c34=value(C34); c45=value(C45); c56=value(C56); c47=value(C47); c68=value(C68); c69=value(C69); c78=value(C78);
    c910=value(C910); c810=value(C810);            
end
GammannValue=value(Gammann);
save('C_Gamma_30Mar.mat', 'Abm', 'Bbm', 'GammaValue', 'GammannValue',...
    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10',...
    'c123', 'c15', 'c34', 'c45', 'c56', 'c47', 'c68', 'c69', 'c78', ...
    'c910', 'c810');



