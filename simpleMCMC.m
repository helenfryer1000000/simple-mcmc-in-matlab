%% Simple MCMC

function [K] = simpleMCMC

clear all

chain_length = 10;
time_span = [0:0.01:10];
K = zeros(chain_length+1,1);
Kprop = zeros(chain_length,1);
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
a = 2;
b = 0.7;
c = 0.8;
initial = [100 1 0];
options = odeset('RelTol', 1e-5, 'NonNegative', [1 2 3]);
[t,x] = ode45(@(t,x) sire(t,x,a,b,c), time_span, initial, options);
x_data(:,1) = x(:,3);

%% Doing the MCMC inference
variance = 0.1;
K(1) = 3; %Initial proposed value of a

for i = 1:chain_length
    
    msg = sprintf('Chain number %d/%d', i, chain_length);
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
    
    %Run once with original value
    a = K(i);
    [~,x2] = ode45(@(t,x) sire(t,x,a,b,c), time_span, initial, options);
    x_original(:,i) = x2(:,3);
    
    %Generate new value to test
    Kprop(i) = K(i) + (variance*randn(1));
    
    %Run again with proposed value to test
    a = Kprop(i);
    [~,x3] = ode45(@(t,x) sire(t,x,a,b,c), time_span, initial, options);
    x_new(:,i) = x3(:,3);
    
    %Compute linear least squares
    
    ls1(i,1) = sum(((x_data(:,1) - x_original(:,i)).^2)./1); %Data versus accepted
    
    if max(x_new(:,i))>0 %if the new value does not produce just 0's
        
        ls2(i,1) = sum(((x_data(:,1) - x_new(:,i)).^2)./1); %Data versus proposed
        
        %Compute probability of acceptance
        p(i,1) = exp((ls1(i,1)-ls2(i,1))./2); %
        
        %Generate random from uniform between 0 and 1
        random = rand;
        
        if random < p(i,1)
            K(i+1) = Kprop(i); %ACCEPT!!! If it's a better fit, make this runs proposal the next runs original
            YayOrNay(i,1) = 1;
        else
            K(i+1) = K(i); %REJECT!!!! If not then it's the same as this guess
            YayOrNay(i,1) = 0;
        end
        
    else %if new K gives model output of just 0's, reject
        K(i+1) = K(i);
    end
    
end
figure
plot(K)

csvwrite('MCMCoutK.csv',K)
csvwrite('MCMCoutYayOrNay.csv',YayOrNay)
csvwrite('All_values.csv',x_original)



end

function dxdt = sire(t,x,a,b,c)

dxdt = [
    - b * x(2) * x(1) + a * x(1)
    b * x(2) * x(1) - c * x(2)
    c * x(2)
    ];

end

%     for j = 1:10
%         if Kprop(i) < 0
%             Kprop(i) = K(i) + (variance*randn(1));
%         end
%     end
%
%     if Kprop(i) < 0
%         fprintf('Problem - proposed K less than 0 too many times!')
%     end


% figure
% for j = 1:3;
%     subplot(3,1,j);
%     plot(t,x(:,j));
%     hold all
%     if j == 1;
%         title('S')
%     elseif j==2;
%         title('I')
%     elseif j==3;
%         title('R')
%     end
% end

%% Different ways of computing least squares
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