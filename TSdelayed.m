%------------------------------------------------------------------------------
% Filename: TS.m
% 
% To simulate performance of a delayed-action Thompson Sampling policy for
% online learning in a multi-armed bandit setting.
%
% This code takes means of reward distributions of different arms (choices) of
% the multi-armed bandit and time horizon to evaluate the policy over.
% The policy is compared with a logarithmic regret bound.
%
% Reference: https://arxiv.org/abs/1505.00553
%
% Author: Naumaan Nayyar
%
% Date: April 4, 2015
%------------------------------------------------------------------------------

close all
clear all
clc

%parameters
N = 2;  % number of arms
true_mean = [0.1,0.05];  % true mean of reward distribution of different arms
T = 50000;  % time horizon
delay = 100;  % delay in feedback, keep using existing data during this period

%initialization
S = zeros(N,1); %number of successes
F = zeros(N,1); %number of failures
Sknown = zeros(N,1); %known number of successes
Fknown = zeros(N,1); %known number of failures
count = zeros(N,1); %number of plays of an arm
a = zeros(T,1);

for i = 1:T
    if(mod(i,delay) == 0)
        Sknown = S;
        Fknown = F;
    end
    thetha = betarnd(Sknown+1,Fknown+1);
    [~,arm] = max(thetha);
    a(i) = arm;
    count(arm) = count(arm) + 1;
    reward_arm = binornd(1,true_mean(arm));
    if (reward_arm == 1)
        S(arm) = S(arm) + 1;
    else
        F(arm) = F(arm) + 1;
    end
end
count = S + F;

%calculations
cumregretTSd = zeros(T,1);

cumregretTSd(1) = 0.1 - true_mean(a(1));
for i = 2:T
    cumregretTSd(i) = cumregretTSd(i-1) + (0.1 - true_mean(a(i)));
end

x = 1:T;
figure,plot(x,cumregretTSd,'-b',x,40*log(x),'-r');
hleg = legend('Cummulative Regret - TS (delayed)','40*log(t)');
set(hleg, 'Location', 'SouthEast')
ylabel('Regret');
xlabel('Time');
title('Mean Channel Rewards = [0.1 0.05];');
