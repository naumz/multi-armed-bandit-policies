%------------------------------------------------------------------------------
% Filename: e3.m
% 
% To simulate performance of E3 policy for online learning in a multi-armed
% bandit setting.
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

u = [0.1 0.5 0.6 0.9];  % true mean of reward distribution of different arms
iter = 10^6;  % Time horizon to run the simulation (in iterations)
gamma = 100;  % parameter governing length of exploration phase 

N = size(u,2);
a = zeros(iter,1);
n = zeros(N,1);
uest = zeros(1,N);

l = 1; %epoch
slotcur = 1;
cr = zeros(1,N);
for l = 1:20
        l
        n
        uest
    for j = 1:gamma
        cr = cr + binornd(1,u);
        n = n + 1;
        for m = 1:N
            a(slotcur) = m;
            slotcur = slotcur + 1;
        end
    end
    uest = cr/(gamma*l);
    [~,ordered] = sort(uest);
    play = ordered(end);
    for j = 1:2^l
        a(slotcur) = play;
        n(play) = n(play) + 1;
        slotcur = slotcur + 1;
    end
end

bestval = max(u);
cumregret = zeros(1,iter);
cumreward = 0;
for i = 1:slotcur-1
    cumreward = cumreward + u(a(i));
    cumregret(i) = i*bestval - cumreward;
end
x = 1:slotcur-1;

plot(x,cumregret,'-b',x,2*N*gamma*0.1*log(x),'-r');
hleg = legend('Cummulative Regret','2Nvlog(t)');
set(hleg, 'Location', 'SouthEast')
ylabel('Regret');
xlabel('Slots');
title('DSEE - u = [0.8 0.78]');