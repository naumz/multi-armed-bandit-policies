%------------------------------------------------------------------------------
% Filename: ucb.m
% 
% To simulate performance of UCB policy for online learning in a multi-armed
% bandit setting.
%
% This code takes means of reward distributions of different arms (choices) of
% the multi-armed bandit and time horizon to evaluate the policy over.
% The policy is compared with a logarithmic regret bound.
%
% Reference: https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
%
% Author: Naumaan Nayyar
%
% Date: April 4, 2015
%------------------------------------------------------------------------------

close all
clear all
clc

u = [0.1 0.5 0.6 0.9];  % true mean of reward distribution of different arms
T = 2000000;  % Time horizon to run the simulation
split = 100000;  % output simulation log every split-th iteration
N = length(u);

g = zeros(1,N); % ucb index
a = zeros(1,T); % action at a particular time
b = zeros(N,T); % stored history of ucb indices
n = 0;
m = zeros(1,N); % number of plays of an arm

%%
%initialization
cr = binornd(1,u);
n = 1;
m = ones(1,N);
uest = cr./m;
g(1,:) = uest(1,:) + sqrt(2*log(n)./m(1,:));

%%
%start of ucb loop
for i=1:T
     if(mod(i,split) == 0)
         i
         m
         uest
         g
     end
    [maxpl,argpl] = max(g(1,:));
    a(i) = argpl;
	n = n + 1;
    m(1,argpl) = m(1,argpl) + 1;
    cr(1,argpl) = cr(1,argpl) + binornd(1,u(1,argpl));
    uest = cr./m;
    g(1,:) = uest(1,:) + sqrt(2*log(n)./m(1,:));
    b(:,i) = g(1,:)';
end

%calculations
cumregretUCB = zeros(T,1);

cumregretUCB(1) = 0.9 - u(a(1));
for i = 2:T
    cumregretUCB(i) = cumregretUCB(i-1) + (0.9 - u(a(i)));
end

x = 1:T;
%% overlapped plot
% figure,plot(x,cumregretUCB,'-g');
% hleg = legend('Cummulative Regret - TS','40*log(t)','Cummulative Regret - UCB');

%%separate plot
figure,plot(x,cumregretUCB,'-b',x,40*log(x),'-r');
hleg = legend('Cummulative Regret - UCB','40*log(t)');
set(hleg, 'Location', 'SouthEast')
ylabel('Regret');
xlabel('Time');
title('Mean Channel Rewards = [0.9 0.4];');