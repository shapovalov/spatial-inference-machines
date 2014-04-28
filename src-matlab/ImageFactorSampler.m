classdef ImageFactorSampler < SteadyFactorSampler 
  %ImageFactorSampler steady factor sampler for images
  % it only preprocesses the images; the main work is delegated to SteadyFactorSampler
  
  properties (Constant)
    NUM_FOLDS = 4; % for hold-out estimation of previous-iteration labels
    MAXSHIFTPLUSR = 121;
    NUMFTYPES = 25;
    
    nLabels = 21;
  end
    
  properties
    num_jobs
    strels
    ftypes
  end
  
  methods 
    % Constructs the class and calls the constructor of the superclass. Input arguments:
    %   train: train set:
    %     train.unaries: unary features of superpixels, cell array of length num_images
    %     train.gtlabels: ground truth labels of superpixels, cell array of length num_images
    %     train.gtlabels: segmentation maps, cell array of length num_images
    %   test: (optional) test set; the structure is similar to the one of the train set.
    %   num_jobs: (optional) number of parallel training jobs (default: 6)
    function self = ImageFactorSampler(train, test, num_jobs)
      self = self@SteadyFactorSampler(1:ImageFactorSampler.NUMFTYPES);
      
      if ~exist('num_jobs','var')
        num_jobs = 6;
      end
      self.num_jobs = num_jobs;
      
      % sample regions centered at radial distances
      lenghts = [20, 50, 100];
      radii = [0, 10, 20];
      ftypeAngles = pi * (0:0.25:1.9)';
      assert(length(radii)*length(ftypeAngles) + 1 == self.NUMFTYPES);
      self.ftypes = zeros(self.NUMFTYPES, 3);
      self.ftypes(1,:) = [0, 0, 10];
      self.ftypes(2:end,1:2) = repmat([sin(ftypeAngles), cos(ftypeAngles)], length(radii), 1);
      for i=1:length(lenghts)
        self.ftypes((i-1)*length(ftypeAngles) + 2 : i*length(ftypeAngles) + 1, 1:2) = ...
          self.ftypes((i-1)*length(ftypeAngles) + 2 : i*length(ftypeAngles) + 1, 1:2) * lenghts(i);
        self.ftypes((i-1)*length(ftypeAngles) + 2 : i*length(ftypeAngles) + 1, 3) = radii(i);
      end
      self.ftypes = round(self.ftypes);
      
      self.strels = arrayfun(@(r) strel('disk', r), self.ftypes(:,3), 'UniformOutput', false);
      
      fprintf('Prepare training factors\n');
      [self.foldsMould structuredFactors spatialFactors] = ...
        self.prepareFoldsMould(train, self.NUM_FOLDS);
      self.factors = [structuredFactors spatialFactors];

      if nargin >= 2
        fprintf('Prepare test factors\n');
        [self.foldsMouldTest structuredFactors spatialFactors] = ...
          self.prepareFoldsMould(test, 1);
        self.factorsTest = [structuredFactors spatialFactors];
      else
        self.foldsMouldTest = [];
        self.factorsTest = [];
      end
    end
    
    function numFolds = getNumFolds(self)
      numFolds = self.NUM_FOLDS;
    end
    
    
    function setTestData(self, test)
      [self.foldsMouldTest structuredFactors spatialFactors] = ...
        self.prepareFoldsMould(test, 1);
      self.factorsTest = [structuredFactors spatialFactors];
    end
    
    function res = testDataAreSet(self)
      res = ~isempty(self.foldsMouldTest);
    end
    
  end
  
  methods (Access = private)
    function [folds structuredFactors spatialFactors] = ...
        prepareFoldsMould(self, dataset, numFolds)
      
      numLabels = self.nLabels;
      numFtypes = size(self.ftypes, 1);
      
      sceneIdx = 1:length(dataset.unaries);
      foldIdx = mod(sceneIdx-1, numFolds) + 1;
      
      folds = cell(1,numFolds);
      structuredFactors = cell(numFolds, 0);
      
      matlabpool('open', self.num_jobs);
      cleaner = onCleanup(@() matlabpool('close'));
      
      for foldnum = 1:numFolds
        % cellfun and cat if needed
        folds{foldnum}.features = cell2mat(dataset.unaries(foldIdx == foldnum));
        tmpLabels = cell2mat(dataset.gtlabels(foldIdx == foldnum));
        
        folds{foldnum}.labels = zeros(size(tmpLabels, 1), numLabels); 
        folds{foldnum}.labels((1:size(tmpLabels, 1))' + (tmpLabels-1)*size(tmpLabels, 1)) = 1;
      end
      
      % SPATIAL FACTORS
      sm = dataset.segmaps;
      % precompute offsets
      offsets = cell(numFolds,1);
      numInFold = zeros(1, length(sceneIdx));
      for i = sceneIdx
        nsegs = max(sm{i}(:));
        assert(nsegs == size(dataset.unaries{i}, 1));
        offsets{foldIdx(i)}(end+1) = nsegs;
        numInFold(i) = length(offsets{foldIdx(i)});
      end
      offsets = cellfun(@(x) [0, cumsum(x)], offsets, 'UniformOutput', false);
      allSpatialFactors = cell(length(sceneIdx), size(self.ftypes,1));
      ftypes_ = self.ftypes;
      strels_ = self.strels;
      parfor i = 1:length(sceneIdx) % scene indices are consecutive
        fold = foldIdx(i);
        imFct = ImageFactorSampler.makeImgSpatialFactors(...
          ftypes_, strels_, sm{i}, offsets{fold}(numInFold(i)));
                
        % transform struct
        sceneFct = cell(1, numFtypes);
        for fct = 1:numFtypes
          sceneFct{fct} = imFct(fct, :);
        end
        allSpatialFactors(i, :) = sceneFct;
        
%         if mod(i, 100) == 0
%           fprintf('Images done: %d\n', i);
%         end
      end
      spatialFactors = cell(numFolds, size(self.ftypes,1));
      
      for fold = 1:numFolds
        for ftype = 1:numFtypes
          spatialFactors{fold,ftype} = cat(2, allSpatialFactors{foldIdx == fold,ftype});
        end
      end
    end
    
  end
      
  
  methods (Static = true, Access = private)
    
    function imgNodeFactors = makeImgSpatialFactors(ftypes, strels, imSegs, foldOffset)
      nsegs = max(imSegs(:));
      imgNodeFactors = cell(size(ftypes, 1),nsegs);
      for i = 1:nsegs
        imgNodeFactors(:,i) = ...
          ImageFactorSampler.makeNodeSpatialFactors(ftypes, strels, i, imSegs, foldOffset);
      end
    end
    
    
    function nodeFact = makeNodeSpatialFactors(ftypes, strels, nodeNum, imSegs, foldOffset)
      border = ImageFactorSampler.MAXSHIFTPLUSR;
      paddedSegs = false(size(imSegs) + 2*[border, border]);
      paddedSegs(border+1 : border+size(imSegs,1), border+1 : border+size(imSegs,2)) = ...
        imSegs == nodeNum;
      
      numFtypes = size(ftypes, 1);
      nodeFact = cell(numFtypes, 1);
      for ftype = 1:numFtypes
        y0 = border - ftypes(ftype,1);
        x0 = border - ftypes(ftype,2);
        ftypeSegs = paddedSegs(y0+1 : y0+size(imSegs,1), x0+1 : x0+size(imSegs,2));
        ftypeSegs = imdilate(ftypeSegs, strels{ftype});
        segnums = imSegs(ftypeSegs);
        seghist = histc(segnums, 1:max(segnums));   % ingnore 0s
        neighnums = find(seghist > 0);
        
        % we use small data types to fit that in memory
        nodeFact{ftype}.ftypenum = uint16(ftype); 
        nodeFact{ftype}.mynum = uint32(nodeNum + foldOffset);  % can be int16 for small datasets
        nodeFact{ftype}.neighnums = uint32(neighnums + foldOffset);  % can be int16 for small datasets
        %nodeFact{ftype}.neighweights = double(seghist(seghist > 0));  % not used right now
        nodeFact{ftype}.pwfeatures = [];  % not used
      end
    end
  end
  
end

