function [num_iters, bounds_minus_ni] = perceptron_experiment(N, d, num_samples)
% perceptron_experiment: Code for running the perceptron experiment in HW1
% Inputs:  N is the number of training examples
%          d is the dimensionality of each example (before adding the 1 for w0)
%          num_samples is the number of times to repeat the experiment
% Outputs: num_iters is the # of iterations PLA takes for each sample
%          bound_minus_ni is the difference between the theoretical bound
%                         and the actual number of iterations
%          (both the outputs should be num_samples long)

%instantiate size of arrays to store information between repetitions
classifications = zeros(N, 1);
allData = zeros(num_samples, d+2);
iterationsBound = zeros(num_samples,1);

for i = 1:num_samples           % repeat experiment 
    weightVector = [0, rand(1,d)];              % w*
    trainingExamples = [ ones(N, 1), 2*rand(N, d)-1 ];      %generation of training examples
    weightNorm = norm(weightVector);            % ||w*|| used for calculating theoretical bound
    R=0;                                        % R used for calculating theoretical bound 
    rho= abs( dot(weightVector, trainingExamples(1, :)) ); %used for calculating theoretical bound
        
    %Classify each training example and find R and rho for theoretical bound
    for j = 1:N       
        %Classify Training Vector
        dotProduct = dot(weightVector, trainingExamples(j, :)); 
        if(dotProduct<0)
            classifications(j) = -1;
        else
            classifications(j) = 1;
        end
        %Update R if norm of Xj greater than all others
        RCandidate = norm(trainingExamples(j, :));
        if(RCandidate > R)
            R=RCandidate;
        end
        %Update rho if lower
        if( abs(dotProduct) < rho)
            rho = abs(dotProduct);
        end
    end
    
    %Send Data to Learning Algorithm
    data = [ trainingExamples, classifications];
    [w, iterations] = perceptron_learn(data);   %send data to learning algorithm
    allData(i, :) = [w, iterations];
    [weightNorm; R; rho];
    iterationsBound(i) = weightNorm^2 * R^2 / rho^2;
end

num_iters = allData(:,d+2); %last column of data is iterations for each sample
bounds_minus_ni = iterationsBound - num_iters;
histogram(num_iters)
histogram( log(bounds_minus_ni))
end
