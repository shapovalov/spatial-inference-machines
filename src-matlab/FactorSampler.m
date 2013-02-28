classdef FactorSampler < matlab.mixin.Copyable
%FACTORSAMPLER is an interface for factor generator from data.
% If your model is static, i.e. does not perform factor selection,
% please extend the SteadyFactorSampler abstract class rather than this one.
% At training time, it is effectively an iterator over factor types. 
% sample() yields tne next d-factors of a certain type, and returns that 
% factors and factor type ID. The best fitting factor types selected during 
% the sampling process are known as active factor types.
  
  methods (Abstract)
    % Sets test data to infer labels
    setTestData(self, testData) 
    
    % Picks next factor type and collects all the factors belonging to that
    %INPUT:
    % prevFitness: fitness of the previously sampled factor type, may
    %   affect the fasctor type sampled next
    %OUTPUT:
    % factorsPack: cell array of lenght nfolds. If the sampling is finished, 
    %     returns empty array. Otherwise, elements are structs w/ fields:
    %   features: matrix of unary features of variables in the fold;
    %     size: nvars x nfeatures
    %   labels: matrix of ground truth labels of the fold variables
    %     in over-complete representation; size: nvars x nlabels 
    %   factors: cell array of d-factors; each element includes:
    %     ftypenum: corresponding factor type ID
    %     mynum: index of variable within the fold that corresponds to the factor destination
    %     neighnums: indices of variables in factor source
    %     pwfeatures: array of pairwise features for the factor; default: empty
    %     neighweights: weights of the impact of individual source points; default: uniform  
    % ftypenum: sampled factor type ID
    [factorsPack ftypenum] = sample(self, prevFitness)
    
    % For the trained sampler (sample() iteration is finished), returns 
    % the factors of active factor types, i.e. those selected during sampling
    %INPUT:
    % isTest: flags if the test data should be used, not training.
    %   There is typically only one fold in test data
    %OUTPUT:
    % factorsPacks: see the factorsPack in sample(). The only difference is
    %   that factorsPacks{fold}.factors contain d-factors of variable types
    % actFtypenums: array of factor type IDs selected during sampling.
    %   factorsPacks{fold}.factors contain only d-factors of those types
    %FAILS if isTest==true, but training data have not been set
    [factorsPacks actFtypenums] = sampleActiveFactors(self, isTest)
    
    % returns nfolds, the number of training folds
    numFolds = getNumFolds(self)
    % returns array of active factor types' IDs, see doc for sampleActiveFactors()
    actFtypes = getActiveFtypes(self)
    % returns the number of active types, it should be fixed before training
    numFtypes = getNumFtypes(self)
    
    % sets the coefficients for active factor types
    setFtypePowers(self, ftypePowers)
    % gets the coefficients for active factor types
    ftypePowers = getFtypePowers(self)
  end
  
end




