clear all
clc
close
%
gamma_n = -0.5*eye(2);
Nw = 10000; % number of sample trajectories
N = 100; % horizon
A = eye(2); B = eye(2);
AK = 0.5*eye(2);
%
% generate disturbance samples
mu = [0; 0];           % Mean vector (zero mean)
Sigma = 0.05 * eye(2); % Covariance matrix (0.05 * I_2)
for j=1:Nw
    w24{j} = mvnrnd(mu, Sigma, Nw)';
    w25{j} = mvnrnd(mu, Sigma, Nw)';
end
%
EE24 = [];
EE25=[];
for j=1:Nw
    e24{j}(:,1)=[0 0]';
    E24{j}(1) = 0;
    e25{j}(:,1)=[0 0]';
    E25{j}(1) = 0;
    udf{j}(:,1)=[0 0]';
    for i = 1:N
        e24{j}(:,i+1)=AK*e24{j}(:,i)+w24{j}(:,i);
        E24{j}(i+1) = norm(e24{j}(:,i+1),2);
        if i==1
            e25{j}(:,i+1)=A*e25{j}(:,i)+B*udf{j}(:,i)+w25{j}(:,i);
            E25{j}(i+1) = norm(e25{j}(:,i+1),inf);
        else
            udf{j}(:,i)=udf{j}(:,i-1)/2+gamma_n*w25{j}(:,i-1);
            e25{j}(:,i+1)=A*e25{j}(:,i)+B*udf{j}(:,i)+w25{j}(:,i);
            E25{j}(i+1) = norm(e25{j}(:,i+1),inf);
        end
    end
    EE24=[EE24 max(E24{j})];
    EE25=[EE25 max(E25{j})];
end

q24 = quantile(EE24, 0.95)
q25 = quantile(EE25, 0.95)
% compute polyhedron
r = 0.831; % Radius of the circle
nn = 100; % Number of points

% Generate points on the circle
theta = linspace(0, 2*pi, nn)'; % Angles
x = r * cos(theta);
y = r * sin(theta);
points = [x, y]; % 2D points

% Create a Polyhedron (Convex Hull)
P = Polyhedron(points);
P24(1)=P;
for i=1:N-1
    P24(i+1)=plus(Polyhedron((AK*P24(i).V')'),P);
    P24(i+1).minVRep;
end

P25=Polyhedron([eye(2);-eye(2)],q25*ones(4,1));
figure; hold on
P24(end).plot('alpha',0.,'EdgeColor','red');
P25.plot('alpha',0.,'EdgeColor','blue')
%
for j=1:10
     plot(e24{j}(1,:),e24{j}(2,:),'*','color','red')
     plot(e25{j}(1,:),e25{j}(2,:),'*','color','blue')
end



    

