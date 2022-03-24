%% Simple MCMC

function [a] = simpleMCMC2

clear all
write = 0;
plot = 0;

chain_length = 2500;
time_span = [0:0.01:10];
a = zeros(chain_length+1,1);
a_prop = zeros(chain_length,1);
b_prop = zeros(chain_length,1);
ls1 = zeros(chain_length,1);
ls2 = zeros(chain_length,1);
p = zeros(chain_length,1);
x_original = zeros(length(time_span),chain_length);
x_new = zeros(length(time_span),chain_length);
YayOrNay = zeros(chain_length,1);
out = zeros(chain_length,1);
out2 = zeros(chain_length,1);
reverseStr = '';

%% Generating "data" for fitting to
A = 2;
B = 0.7;
C = 0.8;
initial = [100 1 0];
options = odeset('RelTol', 1e-5, 'NonNegative', [1 2 3]);
[t,x] = ode45(@(t,x) sire(t,x,A,B,C), time_span, initial, options);
x_data(:,1) = x(:,3);

%% Doing the MCMC inference
variance = 0.05;
a_accepted(1) = 4; %Initial proposed value of a
b_accepted(1) = 2.3;
c_accepted(1) = 5;

for i = 1:chain_length
    
    msg = sprintf('Chain number %d/%d', i, chain_length);
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
    
    %Run once with original value
    a = a_accepted(i);
    b = b_accepted(i);
    c = c_accepted(i);
    
    [~,x2] = ode45(@(t,x) sire(t,x,a,b,c), time_span, initial, options);
    x_original(:,i) = x2(:,3);
    
    %% Generate new value to test, check is >0
    a_prop(i) = a_accepted(i) + (variance*randn(1));
    b_prop(i) = b_accepted(i) + (variance*randn(1));
    c_prop(i) = c_accepted(i) + (variance*randn(1));
    
    for j = 1:10
        if a_prop(i) < 0
            a_prop(i) = a_accepted(i) + (variance*randn(1));
        end
    end
    
    if a_prop(i) < 0
        fprintf('Problem - proposed a less than 0 too many times!')
    end
    
    for k = 1:10
        if b_prop(i) < 0
            b_prop(i) = b_accepted(i) + (variance*randn(1));
        end
    end
    
    if b_prop(i) < 0
        fprintf('Problem - proposed b less than 0 too many times!')
    end
    
    for k = 1:10
        if c_prop(i) < 0
            c_prop(i) = c_accepted(i) + (variance*randn(1));
        end
    end
    
    if c_prop(i) < 0
        fprintf('Problem - proposed c less than 0 too many times!')
    end
    
    %% Run again with proposed value to test
    a = a_prop(i);
    b = b_prop(i);
    c = c_prop(i);
    
    [~,x3] = ode45(@(t,x) sire(t,x,a,b,c), time_span, initial, options);
    x_new(:,i) = x3(:,3);
    
    %% Compute linear least squares, compare, accept or reject
    exp_error = 1; %expected error of measurement
    
    ls1(i,1) = sum(((x_data(:,1) - x_original(:,i)).^2)./exp_error); %Data versus accepted
    
    if max(x_new(:,i))>0 %if the new value does not produce just 0's
        
        ls2(i,1) = sum(((x_data(:,1) - x_new(:,i)).^2)./exp_error); %Data versus proposed
        p(i,1) = exp((ls1(i,1)-ls2(i,1))./2); %Compute probability of acceptance
        random = rand; %Generate random from uniform between 0 and 1
        
        if random < p(i,1)
            a_accepted(i+1) = a_prop(i); %ACCEPT!!! If it's a better fit, make this runs proposal the next runs original
            b_accepted(i+1) = b_prop(i);
            c_accepted(i+1) = c_prop(i);
            
            YayOrNay(i,1) = 1;
        else
            a_accepted(i+1) = a_accepted(i); %REJECT!!!! If not then it's the same as this guess
            b_accepted(i+1) = b_accepted(i);
            c_accepted(i+1) = c_accepted(i);
            YayOrNay(i,1) = 0;
        end
        
    else %if new K gives model output of just 0's, reject
        a_accepted(i+1) = a_accepted(i);
        b_accepted(i+1) = b_accepted(i);
        c_accepted(i+1) = c_accepted(i);
        
        ls2(i,1) = NaN;
    end
    
end

if plot == 1
    plotmaker(x_data,chain_length,x_original,a_accepted,b_accepted,c_accepted,A,B,C)
end

if write == 1
    csvwrite('MCMCoutA.csv',a)
    csvwrite('MCMCoutP.csv',p)
    csvwrite('MCMCoutYayOrNay.csv',YayOrNay)
    csvwrite('All_values.csv',x_original)
end

meanYayorNay = mean(YayOrNay(:,1))
end

function dxdt = sire(t,x,a,b,c)

dxdt = [
    - b * x(2) * x(1) + a * x(1)
    b * x(2) * x(1) - c * x(2)
    c * x(2)
    ];

end

function plotmaker(x_data,chain_length,x_original,a_accepted,b_accepted,c_accepted,A,B,C)
%% Plot mean of all runs (with SD) against data
for k = 1:length(x_data(:,1))
    meanx(k,1) = mean(x_original(k,:));
    sdx(k,1) = std(x_original(k,:));
    lowerbound(k,1) = meanx(k,1) - 1.96*sdx(k,1);
    upperbound(k,1) = meanx(k,1) + 1.96*sdx(k,1);
end

figure
plot(meanx,'b')
hold on
plot(lowerbound,'r')
hold on
plot(upperbound,'r')
hold on
plot(x_data(:,1),'g')
hold on

%% Plot estimated parameters against run number
figure
subplot(3,1,1)
plot(a_accepted)
hold on
plot([0, chain_length],[A A],'--r')
subplot(3,1,2)
plot(b_accepted)
hold on
plot([0, chain_length],[B B],'--r')
subplot(3,1,3)
plot(c_accepted)
hold on
plot([0, chain_length],[C C],'--r')

end



%% Different (worse) ways of computing least squares
%     [~, ~, res] = lsqlin(x_original(:,i),x_data(:,1));
%     SS = sum((res.^2).^0.5);
%     [out(i,1),~] = lsqlin(x_original(:,i),x_data(:,1));
%     [~,resnorm(i)] = lsqlin(x_original(:,i),x_data(:,1));

%         [~, ~, resNEW] = lsqlin(x_new(:,i),x_data(:,1));
%         SS2 = sum((resNEW.^2).^0.5);
%         [out2(i,1),~] = lsqlin(x_new(:,i),x_data(:,1));
%           [~,resnorm2(i)] = lsqlin(x_new(:,i),x_data(:,1));

%% Plotting some shit
% figure
% for j = 1:3;
%     subplot(3,1,j);
%     plot(t,x(:,j));
%     hold all
%     plot(tend,xend(:,j),'-r');
%
%     if j == 1;
%         title('S')
%     elseif j==2;
%         title('I')
%     elseif j==3;
%         title('R')
%     end
% end