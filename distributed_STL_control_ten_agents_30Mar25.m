close all
clc
clear all
%
% system
N = 100;
M = 1e5; % big-M constant
iter=8;
tol=1e-7;
gamma=1e2;
gurobi_iter=10;
dh_ij=5;
dv_ij=2;
% time interval of tasks
t1 = [11 51];  g1 = [N-30 N+1]; j1 = [1 N+1;1 N+1;1 N+1]; % agent-1: joint tasks with 2, 3 and 5
t2 = [11 51]; g2 = [N-30 N+1]; j2 = [1 N+1;1 N+1]; % agent-2: joint tasks with 1 and 3
t3 = [11 51]; g3 = [N-30 N+1]; j3 = [1 N+1;1 N+1;1 N+1]; % agent-3: joint tasks with 1, 2 and 4
t4 = [11 51]; g4 = [N-30 N+1]; j4 = [1 N+1;1 N+1;1 N+1]; % agent-4: joint tasks with 3 and 5 and 7
t5 = [11 51]; g5 = [N-30 N+1]; j5 = [1 N+1;1 N+1;1 N+1]; % agent-5: joint tasks with 1 and 4 and 6
t6 = [11 51]; g6 = [N-30 N+1]; j6 = [1 N+1;1 N+1;1 N+1]; % agent-6: joint tasks with 5 and 8 and 9
t7 = [11 51]; g7 = [N-30 N+1]; j7 = [1 N+1;1 N+1]; % agent-7: joint tasks with 4 and 8
t8 = [11 51]; g8 = [N-30 N+1]; j8 = [1 N+1;1 N+1;1 N+1]; % agent-8: joint tasks with 6 and 7 and 10
t9 = [11 51]; g9 = [N-30 N+1]; j9 = [1 N+1;1 N+1]; % agent-9: joint tasks with 6 and 10
t10= [11 51]; g10 =[N-30 N+1]; j10 =[1 N+1;1 N+1]; % agent-10:joint tasks with 8 and 9
% initial states of the agents
x01=[1.5 10]'; x02=[1.5 15]'; x03=[11.5 10]'; x04=[13.5 10]'; x05=[15.5 10]'; x06=[31.5 10]'; x07=[33.5 10]'; x08=[36.5 10]'; x09=[47 10]'; x010=[47 15]';
% max distance of joint tasks of the agents
prox=1;
e123 = prox; e15 = prox; e34 = prox; e45 = prox; 
e47 = prox; e56 = prox; e68 = prox; e69 = prox;
e78 = prox; e810= prox; e910= prox;
eg=prox;
% input capacity vector (box constraints for all agents)
ulim = 1*[0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8]; % it was 1*[....]
rob=.000*[0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07]; % robustness margin
% agents dynamics
A = eye(2); % [1 0.05;0 .9];
B = eye(2); % [1 .002;0 0.9];
Kgain=diag([-0.5 -0.5]);

%% Tubes
Abar=A+B*Kgain;
theta=0.9997;
num_points = 1e7;
dist = makedist('normal','mu',0,'sigma',0.1);
points= random(dist,2,num_points);
for i=1:num_points
    distances(i)=norm(points(:,i),2);
end
points_with_dist=[distances;points];
sort_bound_index=ceil(num_points*theta);
sorted_points=sortrows(points_with_dist',1); sorted_points=sorted_points';
points_in_CR=sorted_points(2:3,1:sort_bound_index);
%
Ew_theta=Polyhedron(points_in_CR');
Ew_theta.minVRep;
E(1)=Ew_theta;
for i=1:99
    E(i+1)=plus(Polyhedron((Abar*E(i).V')'),Ew_theta);
    E(i+1).minVRep;
end
%
% define Workspace X
X=Polyhedron([0 0;50 0;0 20;50 20]);
Xtight=Polyhedron(X.A,X.b+[min_max(-X.A(1,:),E(end),-1);min_max(-X.A(2,:),E(end),-1);min_max(-X.A(3,:),E(end),-1);min_max(-X.A(4,:),E(end),-1)]);
% define obstacle and regions of interest
O1 = Polyhedron([3 4;9 4;9 6;3 6])+[0;5];
O2 = plus(O1, [18;0]);
O3 = plus(O2, [18;0]);
T1 = Polyhedron([1.5 3;0 1.5;1.5 0;3 1.5]);% T1=plus(T1,[.5;.5]);
G1 = Polyhedron([0 8;2 8;2 10;0 10]); G1=plus(G1,[.5;10]);
T2 = plus(T1,[dh_ij;0]); G2=plus(G1,[dh_ij;0]);
T3 = plus(T2,[dh_ij;0]); G3=plus(G2,[dh_ij;0]);
T4 = plus(T3,[dh_ij;0]); G4=plus(G3,[dh_ij;0]);
T5 = plus(T4,[dh_ij;0]); G5=plus(G4,[dh_ij;0]);
T6 = plus(T5,[dh_ij;0]); G6=plus(G5,[dh_ij;0]);
T7 = plus(T6,[dh_ij;0]); G7=plus(G6,[dh_ij;0]);
T8 = plus(T7,[dh_ij;0]); G8=plus(G7,[dh_ij;0]);
T9 = plus(T8,[dh_ij;0]); G9=plus(G8,[dh_ij;0]);
T10 = plus(T9,[dh_ij;0]); G10=plus(G9,[dh_ij;0]);
%
% plot workspace, RoI and obstacles
figure(1);
X.plot('alpha',0,'LineStyle','--'); hold on;
Xtight.plot('alpha',0.2,'color','lightgray')
O1.plot('color','red'); hold on;
O2.plot('color', 'red');
O3.plot('color', 'red');
T1.plot('color',[0 0.5 0.5]); G1.plot('color',[0 0.5 0.5]);     % teal color
T2.plot('color',[0 1 0]); G2.plot('color',[0 1 0]);             % green color
T3.plot('color',[0 0 1]); G3.plot('color',[0 0 1]);             % blue color
T4.plot('color',[0 1 1]); G4.plot('color',[0 1 1]);             % cyan color
T5.plot('color', [1 0 1]); G5.plot('color', [1 0 1]);           % magenta color
T6.plot('color',[1 1 0]); G6.plot('color',[1 1 0]);             % yellow color
T7.plot('color',[1 0.5 0]); G7.plot('color',[1 0.5 0]);         % orange color
T8.plot('color',[0.5 0 0.5]); G8.plot('color',[0.5 0 0.5]);     % purple color
T9.plot('color',[1 0.8 0.8]); G9.plot('color',[1 0.8 0.8]);     % pink color
T10.plot('color',[0.64 0.16 0.16]); G10.plot('color',[0.64 0.16 0.16]); % brown color
axis([-1 50 -1 22]);
hold off;
%
% output matrix
C1 = [-O1.A;-O2.A;-O3.A;-T1.A;-G1.A];
c1 = [O1.b;O2.b;O3.b;T1.b;G1.b];
C2 = [-O1.A;-O2.A;-O3.A;-T2.A;-G2.A];
c2 = [O1.b;O2.b;O3.b;T2.b;G2.b];
C3 = [-O1.A;-O2.A;-O3.A;-T3.A;-G3.A];
c3 = [O1.b;O2.b;O3.b;T3.b;G3.b];
C4 = [-O1.A;-O2.A;-O3.A;-T4.A;-G4.A];
c4 = [O1.b;O2.b;O3.b;T4.b;G4.b];
C5 = [-O1.A;-O2.A;-O3.A;-T5.A;-G5.A];
c5 = [O1.b;O2.b;O3.b;T5.b;G5.b];
C6 = [-O1.A;-O2.A;-O3.A;-T6.A;-G6.A];
c6 = [O1.b;O2.b;O3.b;T6.b;G6.b];
C7 = [-O1.A;-O2.A;-O3.A;-T7.A;-G7.A];
c7 = [O1.b;O2.b;O3.b;T7.b;G7.b];
C8 = [-O1.A;-O2.A;-O3.A;-T8.A;-G8.A];
c8 = [O1.b;O2.b;O3.b;T8.b;G8.b];
C9 = [-O1.A;-O2.A;-O3.A;-T9.A;-G9.A];
c9 = [O1.b;O2.b;O3.b;T9.b;G9.b];
C10 = [-O1.A;-O2.A;-O3.A;-T10.A;-G10.A];
c10 = [O1.b;O2.b;O3.b;T10.b;G10.b];
%
%% Compute tightening parameters
tighteningO1=zeros(4,1);
tighteningO2=zeros(4,1);
tighteningO3=zeros(4,1);
tighteningT1=zeros(4,1); tighteningG1=zeros(4,1);
tighteningT2=zeros(4,1); tighteningG2=zeros(4,1);
tighteningT3=zeros(4,1); tighteningG3=zeros(4,1);
tighteningT4=zeros(4,1); tighteningG4=zeros(4,1);
tighteningT5=zeros(4,1); tighteningG5=zeros(4,1);
tighteningT6=zeros(4,1); tighteningG6=zeros(4,1);
tighteningT7=zeros(4,1); tighteningG7=zeros(4,1);
tighteningT8=zeros(4,1); tighteningG8=zeros(4,1);
tighteningT9=zeros(4,1); tighteningG9=zeros(4,1);
tighteningT10=zeros(4,1); tighteningG10=zeros(4,1);
%
for i=1:4
    %O1
    tighteningO1(i)=min_max(C1(i,:),E(end),1);
    %O2
    tighteningO2(i)=min_max(C1(i+4,:),E(end),1);
    %O3
    tighteningO3(i)=min_max(C1(i+8,:),E(end),1);
    %T1, G1
    tighteningT1(i)=min_max(C1(i+12,:),E(end),-1);
    tighteningG1(i)=min_max(C1(i+16,:),E(end),-1);
     %T2, G2
    tighteningT2(i)=min_max(C2(i+12,:),E(end),-1);
    tighteningG2(i)=min_max(C2(i+16,:),E(end),-1);
     %T3, G3
    tighteningT3(i)=min_max(C3(i+12,:),E(end),-1);
    tighteningG3(i)=min_max(C3(i+16,:),E(end),-1);
     %T4, G4
    tighteningT4(i)=min_max(C4(i+12,:),E(end),-1);
    tighteningG4(i)=min_max(C4(i+16,:),E(end),-1);
     %T5, G5
    tighteningT5(i)=min_max(C5(i+12,:),E(end),-1);
    tighteningG5(i)=min_max(C5(i+16,:),E(end),-1);
     %T6, G6
    tighteningT6(i)=min_max(C6(i+12,:),E(end),-1);
    tighteningG6(i)=min_max(C6(i+16,:),E(end),-1);
     %T7, G7
    tighteningT7(i)=min_max(C7(i+12,:),E(end),-1);
    tighteningG7(i)=min_max(C7(i+16,:),E(end),-1);
     %T8, G8
    tighteningT8(i)=min_max(C8(i+12,:),E(end),-1);
    tighteningG8(i)=min_max(C8(i+16,:),E(end),-1);
     %T9, G9
    tighteningT9(i)=min_max(C9(i+12,:),E(end),-1);
    tighteningG9(i)=min_max(C9(i+16,:),E(end),-1);
     %T10, G10
    tighteningT10(i)=min_max(C10(i+12,:),E(end),-1);
    tighteningG10(i)=min_max(C10(i+16,:),E(end),-1);
end
%
%% tight polyhedra
cc1 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T1.b+tighteningT1;G1.b+tighteningG1];
cc2 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T2.b+tighteningT2;G2.b+tighteningG2];
cc3 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T3.b+tighteningT3;G3.b+tighteningG3];
cc4 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T4.b+tighteningT4;G4.b+tighteningG4];
cc5 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T5.b+tighteningT5;G5.b+tighteningG5];
cc6 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T6.b+tighteningT6;G6.b+tighteningG6];
cc7 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T7.b+tighteningT7;G7.b+tighteningG7];
cc8 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T8.b+tighteningT8;G8.b+tighteningG8];
cc9 = [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T9.b+tighteningT9;G9.b+tighteningG9];
cc10= [O1.b+tighteningO1;O2.b+tighteningO2;O3.b+tighteningO3;T10.b+tighteningT10;G10.b+tighteningG10];
%cc1=c1;cc2=c2;cc3=c3;cc14=c4;cc5=c5;cc6=c6;cc7=c7;cc8=c8;cc9=c9;cc10=c10;
%
%%
% cost function
Lu = [];
%
% start iterations
tic;
for k=1:iter
    k
    if k==1
        [uk1{k}, xk1{k}, elapsedTime1{k}] = compute_10_agents_STL_three_edges(k,x01,zeros(2,N+1),C1,cc1,rob(1),zeros(2,N+1),zeros(2,N+1),zeros(2,N+1),eg,eg,eg,M,A,B,ulim(1),t1,g1,j1,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk2{k}, xk2{k}, elapsedTime2{k}] = compute_10_agents_STL_two_edges(k,x02,zeros(2,N+1),C2,cc2,rob(2),zeros(2,N+1),zeros(2,N+1),eg,eg,M,A,B,ulim(2),t2,g2,j2,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk3{k}, xk3{k}, elapsedTime3{k}] = compute_10_agents_STL_three_edges(k,x03,zeros(2,N+1),C3,cc3,rob(3),zeros(2,N+1),zeros(2,N+1),zeros(2,N+1),eg,eg,eg,M,A,B,ulim(3),t3,g3,j3,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk4{k}, xk4{k}, elapsedTime4{k}] = compute_10_agents_STL_three_edges(k,x04,zeros(2,N+1),C4,cc4,rob(4),zeros(2,N+1),zeros(2,N+1),zeros(2,N+1),eg,eg,eg,M,A,B,ulim(4),t4,g4,j4,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk5{k}, xk5{k}, elapsedTime5{k}] = compute_10_agents_STL_three_edges(k,x05,zeros(2,N+1),C5,cc5,rob(5),zeros(2,N+1),zeros(2,N+1),zeros(2,N+1),eg,eg,eg,M,A,B,ulim(5),t5,g5,j5,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk6{k}, xk6{k}, elapsedTime6{k}] = compute_10_agents_STL_three_edges(k,x06,zeros(2,N+1),C6,cc6,rob(6),zeros(2,N+1),zeros(2,N+1),zeros(2,N+1),eg,eg,eg,M,A,B,ulim(6),t6,g6,j6,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk7{k}, xk7{k}, elapsedTime7{k}] = compute_10_agents_STL_two_edges(k,x07,zeros(2,N+1),C7,cc7,rob(7),zeros(2,N+1),zeros(2,N+1),eg,eg,M,A,B,ulim(7),t7,g7,j7,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk8{k}, xk8{k}, elapsedTime8{k}] = compute_10_agents_STL_three_edges(k,x08,zeros(2,N+1),C8,cc8,rob(8),zeros(2,N+1),zeros(2,N+1),zeros(2,N+1),eg,eg,eg,M,A,B,ulim(8),t8,g8,j8,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk9{k}, xk9{k}, elapsedTime9{k}] = compute_10_agents_STL_two_edges(k,x09,zeros(2,N+1),C9,cc9,rob(9),zeros(2,N+1),zeros(2,N+1),eg,eg,M,A,B,ulim(9),t9,g9,j9,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        [uk10{k}, xk10{k}, elapsedTime10{k}] = compute_10_agents_STL_two_edges(k,x010,zeros(2,N+1),C10,cc10,rob(10),zeros(2,N+1),zeros(2,N+1),eg,eg,M,A,B,ulim(10),t10,g10,j10,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
        %
    else
        if mod(k,3)==0
            [uk1{k}, xk1{k}, elapsedTime1{k}] = compute_10_agents_STL_three_edges(k,x01,xk1{k-1},C1,cc1,rob(1),xk2{k-1},xk3{k-1},xk5{k-1},eg,eg,eg,M,A,B,ulim(1),t1,g1,j1,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk2{k}=uk2{k-1}; xk2{k}=xk2{k-1}; elapsedTime2{k}=0;
            uk3{k}=uk3{k-1}; xk3{k}=xk3{k-1}; elapsedTime3{k}=0;
            [uk4{k}, xk4{k}, elapsedTime4{k}] = compute_10_agents_STL_three_edges(k,x04,xk4{k-1},C4,cc4,rob(4),xk3{k-1},xk5{k-1},xk7{k-1},eg,eg,eg,M,A,B,ulim(4),t4,g4,j4,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk5{k}=uk5{k-1}; xk5{k}=xk5{k-1}; elapsedTime5{k}=0;
            [uk6{k}, xk6{k}, elapsedTime6{k}] = compute_10_agents_STL_three_edges(k,x06,xk6{k-1},C6,cc6,rob(6),xk5{k-1},xk8{k-1},xk9{k-1},eg,eg,eg,M,A,B,ulim(6),t6,g6,j6,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk7{k}=uk7{k-1}; xk7{k}=xk7{k-1}; elapsedTime7{k}=0;
            uk8{k}=uk8{k-1}; xk8{k}=xk8{k-1}; elapsedTime8{k}=0;
            uk9{k}=uk9{k-1}; xk9{k}=xk9{k-1}; elapsedTime9{k}=0;
            [uk10{k}, xk10{k}, elapsedTime10{k}] = compute_10_agents_STL_two_edges(k,x010,xk10{k-1},C10,cc10,rob(10),xk8{k-1},xk9{k-1},eg,eg,M,A,B,ulim(10),t10,g10,j10,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            %
        elseif mod(k,3)==1
            uk1{k}=uk1{k-1}; xk1{k}=xk1{k-1}; elapsedTime1{k}=0;
            uk2{k}=uk2{k-1}; xk2{k}=xk2{k-1}; elapsedTime2{k}=0;
            [uk3{k}, xk3{k}, elapsedTime3{k}] = compute_10_agents_STL_three_edges(k,x03,xk3{k-1},C3,cc3,rob(3),xk1{k-1},xk2{k-1},xk4{k-1},eg,eg,eg,M,A,B,ulim(3),t3,g3,j3,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk4{k}=uk4{k-1}; xk4{k}=xk4{k-1}; elapsedTime4{k}=0;
            [uk5{k}, xk5{k}, elapsedTime5{k}] = compute_10_agents_STL_three_edges(k,x05,xk5{k-1},C5,cc5,rob(5),xk1{k-1},xk4{k-1},xk6{k-1},eg,eg,eg,M,A,B,ulim(5),t5,g5,j5,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk6{k}=uk6{k-1}; xk6{k}=xk6{k-1}; elapsedTime6{k}=0;
            [uk7{k}, xk7{k}, elapsedTime7{k}] = compute_10_agents_STL_two_edges(k,x07,xk7{k-1},C7,cc7,rob(7),xk4{k-1},xk8{k-1},eg,eg,M,A,B,ulim(7),t7,g7,j7,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk8{k}=uk8{k-1}; xk8{k}=xk8{k-1}; elapsedTime8{k}=0;
            [uk9{k}, xk9{k}, elapsedTime9{k}] = compute_10_agents_STL_two_edges(k,x09,xk9{k-1},C9,cc9,rob(9),xk6{k-1},xk10{k-1},eg,eg,M,A,B,ulim(9),t9,g9,j9,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk10{k}=uk10{k-1}; xk10{k}=xk10{k-1}; elapsedTime10{k}=0;
            %
        else
            uk1{k}=uk1{k-1}; xk1{k}=xk1{k-1}; elapsedTime1{k}=0;
            [uk2{k}, xk2{k}, elapsedTime2{k}] = compute_10_agents_STL_two_edges(k,x02,xk2{k-1},C2,cc2,rob(2),xk1{k-1},xk3{k-1},eg,eg,M,A,B,ulim(2),t2,g2,j2,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk3{k}=uk3{k-1}; xk3{k}=xk3{k-1}; elapsedTime3{k}=0;
            uk4{k}=uk4{k-1}; xk4{k}=xk4{k-1}; elapsedTime4{k}=0;
            [uk5{k}, xk5{k}, elapsedTime5{k}] = compute_10_agents_STL_three_edges(k,x05,xk5{k-1},C5,cc5,rob(5),xk1{k-1},xk4{k-1},xk6{k-1},eg,eg,eg,M,A,B,ulim(5),t5,g5,j5,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk6{k}=uk6{k-1}; xk6{k}=xk6{k-1}; elapsedTime6{k}=0;
            uk7{k}=uk7{k-1}; xk7{k}=xk7{k-1}; elapsedTime7{k}=0;
            [uk8{k}, xk8{k}, elapsedTime8{k}] = compute_10_agents_STL_three_edges(k,x08,xk8{k-1},C8,cc8,rob(8),xk6{k-1},xk7{k-1},xk10{k-1},eg,eg,eg,M,A,B,ulim(8),t8,g8,j8,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            [uk9{k}, xk9{k}, elapsedTime9{k}] = compute_10_agents_STL_two_edges(k,x09,xk9{k-1},C9,cc9,rob(9),xk6{k-1},xk10{k-1},eg,eg,M,A,B,ulim(9),t9,g9,j9,N,gamma,gurobi_iter,E(end),Xtight,Kgain);
            uk10{k}=uk10{k-1}; xk10{k}=xk10{k-1}; elapsedTime10{k}=0;
            %
        end
    end
    % compute robustness for kth iteration
    [rob1{k}, rob_ind1{k}, rob_joint1{k}] = compute_robustness_three_edges(xk1{k},xk2{k},xk3{k},xk5{k},C1,cc1,eg,eg,eg,t1,g1,j1,N);
    [rob2{k}, rob_ind2{k}, rob_joint2{k}] = compute_robustness_two_edges(xk2{k},xk1{k},xk3{k},C2,cc2,eg,eg,t2,g2,j2,N);
    [rob3{k}, rob_ind3{k}, rob_joint3{k}] = compute_robustness_three_edges(xk3{k},xk1{k},xk2{k},xk4{k},C3,cc3,eg,eg,eg,t3,g3,j3,N);
    [rob4{k}, rob_ind4{k}, rob_joint4{k}] = compute_robustness_three_edges(xk4{k},xk3{k},xk5{k},xk7{k},C4,cc4,eg,eg,eg,t4,g4,j4,N);
    [rob5{k}, rob_ind5{k}, rob_joint5{k}] = compute_robustness_three_edges(xk5{k},xk1{k},xk4{k},xk6{k},C5,cc5,eg,eg,eg,t5,g5,j5,N);
    [rob6{k}, rob_ind6{k}, rob_joint6{k}] = compute_robustness_three_edges(xk6{k},xk5{k},xk8{k},xk9{k},C6,cc6,eg,eg,eg,t6,g6,j6,N);
    [rob7{k}, rob_ind7{k}, rob_joint7{k}] = compute_robustness_two_edges(xk7{k},xk4{k},xk8{k},C7,cc7,eg,eg,t7,g7,j7,N);
    [rob8{k}, rob_ind8{k}, rob_joint8{k}] = compute_robustness_three_edges(xk8{k},xk6{k},xk7{k},xk10{k},C8,cc8,eg,eg,eg,t8,g8,j8,N);
    [rob9{k}, rob_ind9{k}, rob_joint9{k}] = compute_robustness_two_edges(xk9{k},xk6{k},xk10{k},C9,cc9,eg,eg,t9,g9,j9,N);
    [rob10{k}, rob_ind10{k}, rob_joint10{k}] = compute_robustness_two_edges(xk10{k},xk8{k},xk9{k},C10,cc10,eg,eg,t10,g10,j10,N);
    %
    [rob1{k} rob2{k} rob3{k} rob4{k} rob5{k} rob6{k} rob7{k} rob8{k} rob9{k} rob10{k}]
    if min([rob1{k} rob2{k} rob3{k} rob4{k} rob5{k} rob6{k} rob7{k} rob8{k} rob9{k} rob10{k}])>=tol
        save('computed_traj_1Mar24.mat','xk1','uk1','xk2','uk2','xk3','uk3','xk4','uk4','xk5','uk5','xk6','uk6','xk7','uk7','xk8','uk8','xk9','uk9','xk10','uk10');
        break
    end
    Lu(k)=norm([uk1{k};uk2{k};uk3{k}],1);
end
time_ten_agents=toc;
        
%
% plot agents trajectories
figure(1); hold on;
plot(xk1{end}(1,:),xk1{end}(2,:),'.-','linewidth',1.5, 'Color',[0 0.5 0.5])
plot(xk2{end}(1,:),xk2{end}(2,:),'.-','linewidth',1.5, 'Color',[0 1 0])
plot(xk3{end}(1,:),xk3{end}(2,:),'.-','linewidth',1.5, 'Color',[0 0 1])
plot(xk4{end}(1,:),xk4{end}(2,:),'.-','linewidth',1.5, 'Color',[0 1 1])
plot(xk5{end}(1,:),xk5{end}(2,:),'.-','linewidth',1.5, 'Color',[1 0 1])
plot(xk6{end}(1,:),xk6{end}(2,:),'.-','linewidth',1.5, 'Color',[1 1 0])
plot(xk7{end}(1,:),xk7{end}(2,:),'.-','linewidth',1.5, 'Color',[1 0.5 0])
plot(xk8{end}(1,:),xk8{end}(2,:),'.-','linewidth',1.5, 'Color',[0.5 0 0.5])
plot(xk9{end}(1,:),xk9{end}(2,:),'.-','linewidth',1.5, 'Color',[1 0.8 0.8])
plot(xk10{end}(1,:),xk10{end}(2,:),'.-','linewidth',1.5, 'Color',[0.64 0.16 0.16])

figure(1); hold on;
plot(x01(1), x01(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[0 0.5 0.5])
plot(x02(1), x02(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[0 1 0])
plot(x03(1), x03(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[0 0 1])
plot(x04(1), x04(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[0 1 1])
plot(x05(1), x05(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[1 0 1])
plot(x06(1), x06(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[1 1 0])
plot(x07(1), x07(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[1 0.5 0])
plot(x08(1), x08(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[0.5 0 0.5])
plot(x09(1), x09(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[1 0.8 0.8])
plot(x010(1), x010(2), 'x', 'MarkerSize', 13,'linewidth',1.5, 'Color',[0.64 0.16 0.16])
%
% plot robustness
figure(2); hold on;
plot([rob1{:}],'Linewidth',1.5);plot([rob2{:}],'Linewidth',1.5);plot([rob3{:}],'Linewidth',1.5);
plot([rob4{:}],'Linewidth',1.5);plot([rob5{:}],'Linewidth',1.5);plot([rob6{:}],'Linewidth',1.5);
plot([rob7{:}],'Linewidth',1.5);plot([rob8{:}],'Linewidth',1.5);plot([rob9{:}],'Linewidth',1.5);
plot([rob10{:}],'Linewidth',1.5);
grid on; hold off;
%
%
% compute robustness
% robustness for agents with one edge and one clique
function [rob, rob_ind, rob_joint] = compute_robustness_one_edge_one_clique(x,xk1,xk2,xk3,C,c,e1,e2,t,g,j,N)
%
robO1=[]; robO2=[]; robO3=[]; robO=[];
robT=[];
robG=[];
rob_1=[];
rob_2=[];
for i=1:N+1
    robO1(i)=min([C(1,:)*x(:,i)+c(1) C(2,:)*x(:,i)+c(2) C(3,:)*x(:,i)+c(3) C(4,:)*x(:,i)+c(4)]);
    robO2(i)=min([C(5,:)*x(:,i)+c(5) C(6,:)*x(:,i)+c(6) C(7,:)*x(:,i)+c(7) C(8,:)*x(:,i)+c(8)]);
    robO3(i)=min([C(9,:)*x(:,i)+c(9) C(10,:)*x(:,i)+c(10) C(11,:)*x(:,i)+c(11) C(12,:)*x(:,i)+c(12)]);
end
rob11=-max(robO1); rob12=-max(robO2); rob13=-max(robO3);
rob1=min([rob11 rob12 rob13]);
%
for i=t(1):t(2)
    robT(i-t(1)+1)=min([C(13,:)*x(:,i)+c(13) C(14,:)*x(:,i)+c(14) C(15,:)*x(:,i)+c(15) C(16,:)*x(:,i)+c(16)]);
end
rob2=max(robT);
for i=g(1):g(2)
    robG(i-g(1)+1)=min([C(17,:)*x(:,i)+c(17) C(18,:)*x(:,i)+c(18) C(19,:)*x(:,i)+c(19) C(20,:)*x(:,i)+c(20)]);
end
rob3=max(robG);
rob_ind=min([rob1 rob2 rob3]);
%
for i=j(1,1):j(1,2)
    rob_1(i-j(1,1)+1)=-norm([x(:,i)-xk1(:,i);x(:,i)-xk2(:,i)],1)+e1;
end
for i=j(2,1):j(2,2)
    rob_2(i-j(2,1)+1)=-norm(x(:,i)-xk3(:,i),1)+e2;
end
rob_joint=min([max(rob_1) max(rob_2)]);
%
rob=min(rob_ind,rob_joint);
%
end
%
% robustness for agents with one clique
function [rob, rob_ind, rob_joint] = compute_robustness_one_clique(x,xk1,xk2,C,c,e1,t,g,j,N)
%
robO1=[]; robO2=[]; robO3=[]; robO=[];
robT=[];
robG=[];
rob_1=[];
rob_2=[];
for i=1:N+1
    robO1(i)=min([C(1,:)*x(:,i)+c(1) C(2,:)*x(:,i)+c(2) C(3,:)*x(:,i)+c(3) C(4,:)*x(:,i)+c(4)]);
    robO2(i)=min([C(5,:)*x(:,i)+c(5) C(6,:)*x(:,i)+c(6) C(7,:)*x(:,i)+c(7) C(8,:)*x(:,i)+c(8)]);
    robO3(i)=min([C(9,:)*x(:,i)+c(9) C(10,:)*x(:,i)+c(10) C(11,:)*x(:,i)+c(11) C(12,:)*x(:,i)+c(12)]);
end
rob11=-max(robO1); rob12=-max(robO2); rob13=-max(robO3);
rob1=min([rob11 rob12 rob13]);
%
for i=t(1):t(2)
    robT(i-t(1)+1)=min([C(13,:)*x(:,i)+c(13) C(14,:)*x(:,i)+c(14) C(15,:)*x(:,i)+c(15) C(16,:)*x(:,i)+c(16)]);
end
rob2=max(robT);
for i=g(1):g(2)
    robG(i-g(1)+1)=min([C(17,:)*x(:,i)+c(17) C(18,:)*x(:,i)+c(18) C(19,:)*x(:,i)+c(19) C(20,:)*x(:,i)+c(20)]);
end
rob3=max(robG);
rob_ind=min([rob1 rob2 rob3]);
%
for i=j(1,1):j(1,2)
    rob_1(i-j(1,1)+1)=-norm([x(:,i)-xk1(:,i);x(:,i)-xk2(:,i)],1)+e1;
end
% for i=j(2,1):j(2,2)
%     rob_2(i-j(2,1)+1)=-norm(x(:,i)-xk3(:,i),1)+e2;
% end
rob_joint=min([max(rob_1)]);
%
rob=min(rob_ind,rob_joint);
%
end
%
% robustness for agents with two edges
function [rob, rob_ind, rob_joint] = compute_robustness_two_edges(x,xk1,xk2,C,c,e1,e2,t,g,j,N)
%
robO1=[]; robO2=[]; robO3=[]; robO=[];
robT=[];
robG=[];
rob_1=[];
rob_2=[];
for i=1:N+1
    robO1(i)=min([C(1,:)*x(:,i)+c(1) C(2,:)*x(:,i)+c(2) C(3,:)*x(:,i)+c(3) C(4,:)*x(:,i)+c(4)]);
    robO2(i)=min([C(5,:)*x(:,i)+c(5) C(6,:)*x(:,i)+c(6) C(7,:)*x(:,i)+c(7) C(8,:)*x(:,i)+c(8)]);
    robO3(i)=min([C(9,:)*x(:,i)+c(9) C(10,:)*x(:,i)+c(10) C(11,:)*x(:,i)+c(11) C(12,:)*x(:,i)+c(12)]);
end
rob11=-max(robO1); rob12=-max(robO2); rob13=-max(robO3);
rob1=min([rob11 rob12 rob13]);
%
for i=t(1):t(2)
    robT(i-t(1)+1)=min([C(13,:)*x(:,i)+c(13) C(14,:)*x(:,i)+c(14) C(15,:)*x(:,i)+c(15) C(16,:)*x(:,i)+c(16)]);
end
rob2=max(robT);
for i=g(1):g(2)
    robG(i-g(1)+1)=min([C(17,:)*x(:,i)+c(17) C(18,:)*x(:,i)+c(18) C(19,:)*x(:,i)+c(19) C(20,:)*x(:,i)+c(20)]);
end
rob3=max(robG);
rob_ind=min([rob1 rob2 rob3]);
%
for i=j(1,1):j(1,2)
    rob_1(i-j(1,1)+1)=-norm(x(:,i)-xk1(:,i),1)+e1;
end
for i=j(2,1):j(2,2)
    rob_2(i-j(2,1)+1)=-norm(x(:,i)-xk2(:,i),1)+e2;
end
rob_joint=min([max(rob_1) max(rob_2)]);
%
rob=min(rob_ind,rob_joint);
%
end
%
% robustness for agents with three edges
function [rob, rob_ind, rob_joint] = compute_robustness_three_edges(x,xk1,xk2,xk3,C,c,e1,e2,e3,t,g,j,N)
%
robO1=[]; robO2=[]; robO3=[]; robO=[];
robT=[];
robG=[];
rob_1=[];
rob_2=[];
rob_3=[];
for i=1:N+1
    robO1(i)=min([C(1,:)*x(:,i)+c(1) C(2,:)*x(:,i)+c(2) C(3,:)*x(:,i)+c(3) C(4,:)*x(:,i)+c(4)]);
    robO2(i)=min([C(5,:)*x(:,i)+c(5) C(6,:)*x(:,i)+c(6) C(7,:)*x(:,i)+c(7) C(8,:)*x(:,i)+c(8)]);
    robO3(i)=min([C(9,:)*x(:,i)+c(9) C(10,:)*x(:,i)+c(10) C(11,:)*x(:,i)+c(11) C(12,:)*x(:,i)+c(12)]);
end
rob11=-max(robO1); rob12=-max(robO2); rob13=-max(robO3);
rob1=min([rob11 rob12 rob13]);
%
for i=t(1):t(2)
    robT(i-t(1)+1)=min([C(13,:)*x(:,i)+c(13) C(14,:)*x(:,i)+c(14) C(15,:)*x(:,i)+c(15) C(16,:)*x(:,i)+c(16)]);
end
rob2=max(robT);
for i=g(1):g(2)
    robG(i-g(1)+1)=min([C(17,:)*x(:,i)+c(17) C(18,:)*x(:,i)+c(18) C(19,:)*x(:,i)+c(19) C(20,:)*x(:,i)+c(20)]);
end
rob3=max(robG);
rob_ind=min([rob1 rob2 rob3]);
%
for i=j(1,1):j(1,2)
    rob_1(i-j(1,1)+1)=-norm(x(:,i)-xk1(:,i),1)+e1;
end
for i=j(2,1):j(2,2)
    rob_2(i-j(2,1)+1)=-norm(x(:,i)-xk2(:,i),1)+e2;
end
for i=j(3,1):j(3,2)
    rob_3(i-j(3,1)+1)=-norm(x(:,i)-xk3(:,i),1)+e3;
end
rob_joint=min([max(rob_1) max(rob_2) max(rob_3)]);
%
rob=min(rob_ind,rob_joint);
%
end


%% tightening functions
function minGamma = findMinGamma(g, theta, time, min_d, max_d,num)
% Pr[ax+b >= 0] <= 1-theta ( or Pr[ax+b<=0] >= theta )
t1 = time(1);
t2 = time(2);
% Number of samples for Monte Carlo simulation
num_samples = num;
dist = makedist('Triangular', 'a', min_d, 'b', 0, 'c', max_d);
%
index = round((1 - theta) * num_samples);
%    
for i =t1:t2
    w_samples=random(dist,2*i,num_samples);
    sorted_samples=sort(g*kron(ones(1,i),eye(2,2))*w_samples);
    gamma(i) = sorted_samples(index);
end
minGamma=gamma;
%
end
%
function maxGamma = findMaxGamma(g, theta, time, min_d, max_d,num)
% Pr[ax+b >= 0] >= theta
t1 = time(1);
t2 = time(2);
% Number of samples for Monte Carlo simulation
num_samples = num;
dist = makedist('Triangular', 'a', min_d, 'b', 0, 'c', max_d);
%
index = round(theta*num_samples);
%    
for i = t1:t2
    w_samples=random(dist,2*i,num_samples);
    sorted_samples=sort(g*kron(ones(1,i),eye(2,2))*w_samples);
    gamma(i) = sorted_samples(index);
end
maxGamma=gamma;
%
end
%
function gamma=min_max(a,E,neg)
%
er=sdpvar(2,1);
F = [E(end).A*er<=E(end).b];
ops = sdpsettings;
obj = -neg*a*er;
optimize(F, obj, ops);
gamma=a*value(er);
%
end























