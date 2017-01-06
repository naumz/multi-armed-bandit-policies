%------------------------------------------------------------------------------
% Filename: ucbdelayed.m
% 
% To simulate performance of a delayed-action UCB policy for online learning
% in a multi-armed bandit setting.
%
% This code takes means of reward distributions of different arms (choices) of
% the multi-armed bandit and time horizon to evaluate the policy over.
% The policy is compared with a logarithmic regret bound.
%
% Reference: http://www-scf.usc.edu/~nnayyar/files/KaNaJa14.pdf
%
% Author: Naumaan Nayyar
%
% Date: April 4, 2015
%------------------------------------------------------------------------------


close all
clear all
clc

u = [0.1 0.05];  % true mean of reward distribution of different arms
T = 50000;  % Time horizon to run the simulation
delay = 1000;  % delay in feedback, keep using existing data during this period
% split = 10000;  % output simulation log every split-th iteration

g = zeros(1,2); %ucb index
a = zeros(1,T); %action at a particular time
b = zeros(2,T); %stored history of ucb indices
n = 0;
m = zeros(1,2); %number of plays of an arm

%%
%initialization
cr = binornd(1,u);
n = 1;
m = ones(1,2);
uest = cr./m;
g(1,:) = uest(1,:) + sqrt(3*log(n)./m(1,:));

%%
%start of ucb loop
for i=1:T
    %     if(mod(i,split) == 0)
    %         i
    %         m
    %         uest
    %         g
    %     end
    [maxpl,argpl] = max(g(1,:));
    if(mod(i,delay) == 1)
        
        a(i) = argpl;
    else
        a(i) = a(i-1);
    end
    n = n + 1;
    m(1,a(i)) = m(1,a(i)) + 1;
    cr(1,a(i)) = cr(1,a(i)) + binornd(1,u(1,a(i)));
    uest = cr./m;
    g(1,:) = uest(1,:) + sqrt(2*log(n)./m(1,:));
    b(:,i) = g(1,:)';
end

%calculations
cumregretUCBd = zeros(T,1);

cumregretUCBd(1) = 0.1 - u(a(1));
for i = 2:T
    cumregretUCBd(i) = cumregretUCBd(i-1) + (0.1 - u(a(i)));
end

x = 1:T;
%%overlapped plot
% hold on
% plot(x,cumregretUCB,'-g');
% hleg = legend('Cummulative Regret - TS','40*log(t)','Cummulative Regret - UCB');

%separate plot
figure,plot(x,cumregretUCBd,'-b',x,40*log(x),'-r');
hleg = legend('Cummulative Regret - UCB (delayed)','40*log(t)');
set(hleg, 'Location', 'SouthEast')
ylabel('Regret');
xlabel('Time');
title('Mean Channel Rewards = [0.9 0.4];');