%------------------------------------------------------------------------------
% Filename: TS.m
% 
% To simulate performance of a Thompson Sampling policy for online learning
% in a multi-armed bandit setting.
%
% This code takes means of reward distributions of different arms (choices) of
% the multi-armed bandit and time horizon to evaluate the policy over.
% The policy is compared with a logarithmic regret bound.
%
% Reference: http://jmlr.org/proceedings/papers/v23/agrawal12/agrawal12.pdf
%
% Author: Naumaan Nayyar
%
% Date: April 4, 2015
%------------------------------------------------------------------------------


clear all
clc

%parameters
N = 2;  % number of arms
true_mean = [0.1,0.5,0.6,0.9];  % true mean of reward distribution of different arms
N = length(true_mean);  % number of arms

T = 2000000;  % Time horizon to run the simulation

%initialization
S = zeros(N,1); %number of successes
F = zeros(N,1); %number of failures
count = zeros(N,1); %number of plays of an arm
a = zeros(T,1);

for i = 1:T
    if (mod(i,10000)==0)
        i
        S
        F
    end
    thetha = betarnd(S+1,F+1);
    [~,arm] = max(thetha);
    a(i) = arm;
    reward_arm = binornd(1,true_mean(arm));
    if (reward_arm == 1)
        S(arm) = S(arm) + 1;
    else
        F(arm) = F(arm) + 1;
    end
end
count = S + F;

%calculations
cumregretTS = zeros(T,1);

cumregretTS(1) = 0.9 - true_mean(a(1));
for i = 2:T
    cumregretTS(i) = cumregretTS(i-1) + (0.9 - true_mean(a(i)));
end

x = 1:T;

%% separate plot
figure,plot(x,cumregretTS,'-b',x,25*log(x),'-r');
hleg = legend('Cummulative Regret - TS','25 log(t)');
set(hleg, 'Location', 'SouthEast')
ylabel('Regret');
xlabel('Time');
title('Mean Channel Rewards = [0.1,0.5,0.6,0.9];');

%% overlapped plot
% hold on
% plot(x,cumregretTS,'-g');
% hleg = legend('Cummulative Regret - TS (delayed)','40*log(t)','Cummulative Regret - TS');

