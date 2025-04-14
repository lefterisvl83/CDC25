function [uopt, xopt, elapsedTime] = compute_10_agents_STL_two_edges(uk_1,k,x0,xk,C,c,rob,xk1,xk2,e1,e2,M,A,B,ulim,t,g,j,N,gamma,gurobi_iter,E,X)% E is the PRS, X is the tight workspace
y1=[];
y2=[];
for i=1:N+1
    y1(i)=-norm(xk(:,i)-xk1(:,i),1)+e1;
    y2(i)=-norm(xk(:,i)-xk2(:,i),1)+e2;
end
robk1=max(y1);
robk2=max(y2);
if robk1>=robk2
    select=2;
else
    select=1;
end
%

% define the variables
yO11=binvar(N+1,1,'full');
yO12=binvar(N+1,1,'full');
yO13=binvar(N+1,1,'full');
yO14=binvar(N+1,1,'full');
%
yO21=binvar(N+1,1,'full');
yO22=binvar(N+1,1,'full');
yO23=binvar(N+1,1,'full');
yO24=binvar(N+1,1,'full');
%
yO31=binvar(N+1,1,'full');
yO32=binvar(N+1,1,'full');
yO33=binvar(N+1,1,'full');
yO34=binvar(N+1,1,'full');
%
yT1=binvar(t(2)-t(1)+1,1,'full');
yT2=binvar(t(2)-t(1)+1,1,'full');
yT3=binvar(t(2)-t(1)+1,1,'full');
yT4=binvar(t(2)-t(1)+1,1,'full');
%
yG1=binvar(g(2)-g(1)+1,1,'full');
yG2=binvar(g(2)-g(1)+1,1,'full');
yG3=binvar(g(2)-g(1)+1,1,'full');
yG4=binvar(g(2)-g(1)+1,1,'full');
%
x1=binvar(j(1,2)-j(1,1)+1,1,'full');
x2=binvar(j(2,2)-j(2,1)+1,1,'full');
%
x = sdpvar(2,N+1,'full');
u = sdpvar(2,N,'full');
mu = sdpvar(1,1,'full');
%
rhoO1=binvar(N+1,1,'full');
rhoO2=binvar(N+1,1,'full');
rhoO3=binvar(N+1,1,'full');
rhoT=binvar(t(2)-t(1)+1,1,'full');
rhoG=binvar(g(2)-g(1)+1,1,'full');
%
r1=binvar(1,1,'full');
r2=binvar(1,1,'full');
rho011=binvar(1,1,'full');
rho11=binvar(1,1,'full');
rho021=binvar(1,1,'full');
rho21=binvar(1,1,'full');
rho031=binvar(1,1,'full');
rho31=binvar(1,1,'full');
rho1=binvar(1,1,'full');
rho2=binvar(1,1,'full');
rho3=binvar(1,1,'full');
rho=binvar(1,1,'full');
%
% constraints
U = Polyhedron([eye(2);-eye(2)],ulim*ones(4,1));
Utight= U;
%
F = [x(:,1)==x0];
if k==1
    for i = 1:N
        F = [F, x(:,i+1)==A*x(:,i)+B*u(:,i)];
        F = [F, X.A*x(:,i+1)<=X.b];
        F = [F, Utight.A*u(:,i)<=Utight.b];
    end
else
    for i = 1:k
        F = [F, x(:,i)==xk(:,i)];
        F = [F, u(:,i)==uk_1(:,i)];
    end
    for i=k:N
        F = [F, x(:,i+1)==A*x(:,i)+B*u(:,i)];
        F = [F, X.A*x(:,i+1)<=X.b];
        F = [F, Utight.A*u(:,i)<=Utight.b];
    end
end

% obstacle avoidance (ALWAYS)
for i = 1:N+1
    F = [F, C(1,:)*x(:,i)+c(1)-M*yO11(i)<=-rob, C(1,:)*x(:,i)+c(1)+M*(1-yO11(i))>=-rob, rhoO1(i)<=yO11(i)];
    F = [F, C(2,:)*x(:,i)+c(2)-M*yO12(i)<=-rob, C(2,:)*x(:,i)+c(2)+M*(1-yO12(i))>=-rob, rhoO1(i)<=yO12(i)];
    F = [F, C(3,:)*x(:,i)+c(3)-M*yO13(i)<=-rob, C(3,:)*x(:,i)+c(3)+M*(1-yO13(i))>=-rob, rhoO1(i)<=yO13(i)];
    F = [F, C(4,:)*x(:,i)+c(4)-M*yO14(i)<=-rob, C(4,:)*x(:,i)+c(4)+M*(1-yO14(i))>=-rob, rhoO1(i)<=yO14(i)];
    F = [F, rhoO1(i)>=1-4+yO11(i)+yO12(i)+yO13(i)+yO14(i)];
    F = [F, rho011>=rhoO1(i)];
end
F = [F, rho011<=sum(rhoO1), rho11==1-rho011];
%
for i = 1:N+1
    F = [F, C(5,:)*x(:,i)+c(5)-M*yO21(i)<=-rob, C(5,:)*x(:,i)+c(5)+M*(1-yO21(i))>=-rob, rhoO2(i)<=yO21(i)];
    F = [F, C(6,:)*x(:,i)+c(6)-M*yO22(i)<=-rob, C(6,:)*x(:,i)+c(6)+M*(1-yO22(i))>=-rob, rhoO2(i)<=yO22(i)];
    F = [F, C(7,:)*x(:,i)+c(7)-M*yO23(i)<=-rob, C(7,:)*x(:,i)+c(7)+M*(1-yO23(i))>=-rob, rhoO2(i)<=yO23(i)];
    F = [F, C(8,:)*x(:,i)+c(8)-M*yO24(i)<=-rob, C(8,:)*x(:,i)+c(8)+M*(1-yO24(i))>=-rob, rhoO2(i)<=yO24(i)];
    F = [F, rhoO2(i)>=1-4+yO21(i)+yO22(i)+yO23(i)+yO24(i)];
    F = [F, rho021>=rhoO2(i)];
end
F = [F, rho021<=sum(rhoO2), rho21==1-rho021];
%
for i = 1:N+1
    F = [F, C(9,:)*x(:,i)+c(9)-M*yO31(i)<=-rob, C(9,:)*x(:,i)+c(9)+M*(1-yO31(i))>=-rob, rhoO3(i)<=yO31(i)];
    F = [F, C(10,:)*x(:,i)+c(10)-M*yO32(i)<=-rob, C(10,:)*x(:,i)+c(10)+M*(1-yO32(i))>=-rob, rhoO3(i)<=yO32(i)];
    F = [F, C(11,:)*x(:,i)+c(11)-M*yO33(i)<=-rob, C(11,:)*x(:,i)+c(11)+M*(1-yO33(i))>=-rob, rhoO3(i)<=yO33(i)];
    F = [F, C(12,:)*x(:,i)+c(12)-M*yO34(i)<=-rob, C(12,:)*x(:,i)+c(12)+M*(1-yO34(i))>=-rob, rhoO3(i)<=yO34(i)];
    F = [F, rhoO3(i)>=1-4+yO31(i)+yO32(i)+yO33(i)+yO34(i)];
    F = [F, rho031>=rhoO3(i)];
end
F = [F, rho031<=sum(rhoO3), rho31==1-rho031];
%
F = [F, rho1<=rho11, rho1<=rho21, rho1<=rho31, rho1>=1-3+rho11+rho21+rho31];
% pass by region T (EVENTUALLY)
for i = t(1):t(2)
    F = [F, C(13,:)*x(:,i)+c(13)-M*yT1(i-t(1)+1)<=rob, C(13,:)*x(:,i)+c(13)+M*(1-yT1(i-t(1)+1))>=rob, rhoT(i-t(1)+1)<=yT1(i-t(1)+1)];
    F = [F, C(14,:)*x(:,i)+c(14)-M*yT2(i-t(1)+1)<=rob, C(14,:)*x(:,i)+c(14)+M*(1-yT2(i-t(1)+1))>=rob, rhoT(i-t(1)+1)<=yT2(i-t(1)+1)];
    F = [F, C(15,:)*x(:,i)+c(15)-M*yT3(i-t(1)+1)<=rob, C(15,:)*x(:,i)+c(15)+M*(1-yT3(i-t(1)+1))>=rob, rhoT(i-t(1)+1)<=yT3(i-t(1)+1)];
    F = [F, C(16,:)*x(:,i)+c(16)-M*yT4(i-t(1)+1)<=rob, C(16,:)*x(:,i)+c(16)+M*(1-yT4(i-t(1)+1))>=rob, rhoT(i-t(1)+1)<=yT4(i-t(1)+1)];
    F = [F, rhoT(i-t(1)+1)>=1-4+yT1(i-t(1)+1)+yT2(i-t(1)+1)+yT3(i-t(1)+1)+yT4(i-t(1)+1)];
    F = [F, rho2>=rhoT(i-t(1)+1)];
end
F = [F, rho2<=sum(rhoT)];
% pass by goal region (EVENTUALLY)
for i = g(1):g(2)
    F = [F, C(17,:)*x(:,i)+c(17)-M*yG1(i-g(1)+1)<=rob, C(17,:)*x(:,i)+c(17)+M*(1-yG1(i-g(1)+1))>=rob, rhoG(i-g(1)+1)<=yG1(i-g(1)+1)];
    F = [F, C(18,:)*x(:,i)+c(18)-M*yG2(i-g(1)+1)<=rob, C(18,:)*x(:,i)+c(18)+M*(1-yG2(i-g(1)+1))>=rob, rhoG(i-g(1)+1)<=yG2(i-g(1)+1)];
    F = [F, C(19,:)*x(:,i)+c(19)-M*yG3(i-g(1)+1)<=rob, C(19,:)*x(:,i)+c(19)+M*(1-yG3(i-g(1)+1))>=rob, rhoG(i-g(1)+1)<=yG3(i-g(1)+1)];
    F = [F, C(20,:)*x(:,i)+c(20)-M*yG4(i-g(1)+1)<=rob, C(20,:)*x(:,i)+c(20)+M*(1-yG4(i-g(1)+1))>=rob, rhoG(i-g(1)+1)<=yG4(i-g(1)+1)];
    F = [F, rhoG(i-g(1)+1)>=1-4+yG1(i-g(1)+1)+yG2(i-g(1)+1)+yG3(i-g(1)+1)+yG4(i-g(1)+1)];
    F = [F, rho3>=rhoG(i-g(1)+1)];
end
F = [F, rho3<=sum(rhoG)];
% Coupling constraints (EVENTUALLY)
if k>1
    if select==1
        F = [F, mu>=min(0,robk1)];
        for i = j(1,1):j(1,2)
            F = [F, -norm(xk1(:,i)-x(:,i),1)+e1-M*x1(i-j(1,1)+1)<=mu, -norm(xk1(:,i)-x(:,i),1)+e1+M*(1-x1(i-j(1,1)+1))>=mu, r1>=x1(i-j(1,1)+1)];
        end
        for i = j(2,1):j(2,2)
            F = [F, -norm(x(:,i)-xk2(:,i),1)+e2-M*x2(i-j(2,1)+1)<=min(0,robk2), -norm(x(:,i)-xk2(:,i),1)+e2+M*(1-x2(i-j(2,1)+1))>=min(0,robk2), r2>=x2(i-j(2,1)+1)];
        end
    else
        F = [F, mu>=min(0,robk2)];
        for i =j(1,1):j(1,2)
            F = [F, -norm(xk1(:,i)-x(:,i),1)+e1-M*x1(i-j(1,1)+1)<=min(0,robk1), -norm(xk1(:,i)-x(:,i),1)+e1+M*(1-x1(i-j(1,1)+1))>=min(0,robk1), r1>=x1(i-j(1,1)+1)];
        end
        for i =j(2,1):j(2,2)
            F = [F, -norm(x(:,i)-xk2(:,i),1)+e2-M*x2(i-j(2,1)+1)<=mu, -norm(x(:,i)-xk2(:,i),1)+e2+M*(1-x2(i-j(2,1)+1))>=mu, r2>=x2(i-j(2,1)+1)];
        end
    end
    F = [F, r1<=sum(x1)];
    F = [F, r2<=sum(x2)];
else
    F =[F, r1==1];
    F = [F, r2==1];
    F = [F, mu==0];
end
% 
F = [F, rho<=rho1, rho<=rho2, rho<=rho3, rho<=r1, rho<=r2];
F = [F, rho>=1-5+rho1+rho2+rho3+r1+r2];
F = [F, rho==1];
%
% optimize
ops = sdpsettings;
% % ops.verbose = 0; % configure level of info solver displays
% ops.solver = 'gurobi'; % choose solver
ops = sdpsettings('solver', 'gurobi','gurobi.SolutionLimit', gurobi_iter);
%
% objective
objective = norm(u,1)-gamma*mu;
%
tic;
optimize(F, objective, ops);
elapsedTime = toc;
% result
uopt = value(u);
xopt = value(x);
%
end