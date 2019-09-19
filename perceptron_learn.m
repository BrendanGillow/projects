function [w, iterations] = perceptron_learn(data_in)
% perceptron_learn: Run PLA on the input data
% Inputs:  data_in is a matrix with each row representing an (x,y) pair;
%                 the x vector is augmented with a leading 1,
%                 the label, y, is in the last column
% Outputs: w is the learned weight vector; 
%            it should linearly separate the data if it is linearly separable
%          iterations is the number of iterations the algorithm ran for

dataInSize = size(data_in);
N = dataInSize(1);      %number of training examples
d = dataInSize(2) -2;   % Dimensionality of samples space Xo added in front and Y at end
w = zeros(1, d+1);      %weight vector to be updated 
nextAdjustment = 0;          %x to adjust by each iteration
iterations =0;          %iterations til w is found

while(nextAdjustment >= 0)
    for x = 1:N
        trueY = data_in(x, d+2);                % actual value of Y
        dotProduct = dot( w, data_in(x, 1:d+1));     %classification by w 
        classY=0;                               %classifed Y value
        if(dotProduct <0)
            classY=-1;
        else
            classY=1;
        end
        if(trueY ~= classY)                     % labels first misclasified point as next adjustment for w
            nextAdjustment = x;
            break;
        end
    end
    if( nextAdjustment ==0)                     % if all examples classified correctly exit
        break;
    end
    w = w + data_in(nextAdjustment, 1:d+1)*data_in(nextAdjustment,d+2);        %adjust w by misclasified vector
    nextAdjustment =0;                          % used for identifying when algorithm is completed
    iterations = iterations +1;                 % keep track of iterations
end
