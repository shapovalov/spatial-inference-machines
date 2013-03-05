classdef SteadyFactorSampler < FactorSampler
%SteadyFactorSampler is a partial implementation of FactorSampler
% where active factor types are known and fixed, not sampled randomly.
% In other words, active factor types are all the types sampled during training.
% To implement it, you only need to implement:
%  * numFolds = getNumFolds(self)
%  * setTestData(self, testData) and testDataAreSet(self)
%  * a constructor that calls SteadyFactorSampler(ftypenums) to provide factor type IDs 
%     and sets the following: foldsMould, factors, foldsMouldTest, factorsTest
  
  properties (SetAccess = private, GetAccess = private)
    ftypenums
    curFtypenum = 1
    ftypePowers
  end
    
  properties (SetAccess = protected, GetAccess = protected)
    % The following two fields should be pre-computed in the constructor.
    
    % foldsMould is a template for folds that contains variables' features 
    % and ground truth labels. When sample() is called, only d-factors of 
    % the corresponding type are added to the copy of foldsMould.
    % See the documentation for FactorSample.sample(), factorPack argument.
    % foldsMould has the same format, but the factors fields are not set.
    foldsMould
    
    % factors is a 2D cell array of size nfolds x nftypes that caches the
    % factors of all (active) factor types. The corresponding columns are
    % used in calls of sample(). Individual elements are cell arrays.
    % See the documentation for FactorSample.sample(), factorPack argument.
    % factors{fold,ftype} has the same format as factorPack.factors
    factors 
    
    % The following two fields should be pre-computed in the constructor,
    % if the test set is provided. They should also be computed in setTestData()
    % The format is the same as for foldsMould/factors fields. Not that it
    % only makes sense to use one fold in test data.
    foldsMouldTest
    factorsTest % 2D cell array of factor struct arrays
  end
  
  methods
    function self = SteadyFactorSampler(ftypenums)
      self.ftypenums = ftypenums;
    end
    
    function actFtypes = getActiveFtypes(self)
      actFtypes = self.ftypenums;
    end
    
    function numFtypes = getNumFtypes(self)
      numFtypes = length(self.ftypenums);
    end
    
    function setFtypePowers(self, ftypePowers)
      self.ftypePowers = ftypePowers;
    end
    
    function ftypePowers = getFtypePowers(self)
      ftypePowers = self.ftypePowers; 
    end
    
    % returns folds for a certain factor
    function [factorsPack ftypenum] = sample(self, ~) % ignore fitness
      if self.curFtypenum > length(self.ftypenums)
        factorsPack = [];
        ftypenum = 0; % anything goes
        return
      end
      
      factorsPack = self.foldsMould;
      for foldnum = 1:self.getNumFolds()
        factorsPack{foldnum}.factors = self.factors{foldnum, self.curFtypenum};
      end
      
      ftypenum = self.ftypenums(self.curFtypenum);
      self.curFtypenum = self.curFtypenum + 1;
    end
    
    function [factorsPacks actFtypenums] = sampleActiveFactors(self, isTest)
      if nargin >= 2 && isTest
        if isempty(self.foldsMouldTest)
          factorsPacks = [];
          return % TODO: just return?
        end  
        factorsPacks = self.foldsMouldTest;
        factorsPacks{1}.factors = {}; 
        for i = 1:length(self.ftypenums)
          factorsPacks{1}.factors = ...
            [factorsPacks{1}.factors self.factorsTest{1,i}];
        end
      else
        factorsPacks = self.foldsMould;
        for foldnum = 1:self.getNumFolds()
          factorsPacks{foldnum}.factors = {}; 
          for i = 1:length(self.ftypenums)
            factorsPacks{foldnum}.factors = ...
              [factorsPacks{foldnum}.factors self.factors{foldnum,i}];
          end
        end
      end
      
      actFtypenums = self.ftypenums;
    end
  end
  
end

