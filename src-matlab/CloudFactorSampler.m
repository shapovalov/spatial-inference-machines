classdef CloudFactorSampler < SteadyFactorSampler 
  %CORNELLCLOUDFACTORSAMPLER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties (Constant)
    NUM_FOLDS = 6;
    
    ftypes = [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1;    % local
              0.1, -0.3, -1000, 2.5, 0.3, -0.1;  % to-restr, down
              -2.5, -0.3, 0.1, -0.1, 0.3, 1000;  % from-restr, up
              -0.3, -0.3, -1000, 0.3, 0.3, -0.1;  % down
              -0.3, -1.0, -0.3, 0.3, -0.1, 0.3;   % left-restr
              -0.3, 0.1, -0.3, 0.3, 1.0, 0.3;     % right-restr
              -2.5, -0.3, -0.3, -0.1, 0.3, 0.3;  % from-restr
              0.1, -0.3, -0.3, 2.5, 0.3, 0.3;    % to-restr
              -0.3, -0.3, 0.1, 0.3, 0.3, 1000;    % up
             ];
  end
    
  properties
    num_jobs
    nLabels
  end
  
  methods 
    function self = CloudFactorSampler(train, test, pcdPathMask, num_jobs)
      self = self@SteadyFactorSampler(0:size(CloudFactorSampler.ftypes,1));
      
      if ~exist('num_jobs','var')
        num_jobs = 6;
      end
      self.num_jobs = num_jobs;
      
      % TODO!!!
      addpath('..\src\PointCloudInfMachine\x64\Release'); % for mex region_segments
      
      self.nLabels = max([train.nodesFtLb(:,3); test.nodesFtLb(:,3)]);
      
      [self.foldsMould structuredFactors spatialFactors] = ...
        self.prepareFoldsMould(train, self.NUM_FOLDS, pcdPathMask);
      self.factors = [structuredFactors spatialFactors];

      if nargin >= 2
        [self.foldsMouldTest structuredFactors spatialFactors] = ...
          self.prepareFoldsMould(test, 1, pcdPathMask);
        self.factorsTest = [structuredFactors spatialFactors];
      end
    end
    
    function numFolds = getNumFolds(self)
      numFolds = self.NUM_FOLDS;
    end
    
    
    function setTestData(self, test, pcdPathMask)
      [self.foldsMouldTest structuredFactors spatialFactors] = ...
        self.prepareFoldsMould(test, 1, pcdPathMask);
      self.factorsTest = [structuredFactors spatialFactors];
    end
    
  end
  
  methods (Access = private)
    function [folds structuredFactors spatialFactors] = ...
        prepareFoldsMould(self, features, numFolds, pcdPathMask)
      %assert(mod(size(features,1), self.NUM_FOLDS) == 0); %TODO: not really
      
      sceneIdx = unique(features.nodesFtLb(:,1));
      
      foldIdxNodes = zeros(1, size(features.nodesFtLb, 1));
      foldIdxEdges = zeros(1, size(features.edgesFtLb, 1));
      for i = 1:length(sceneIdx)
        foldIdxNodes(features.nodesFtLb(:,1) == sceneIdx(i)) = mod(i, numFolds) + 1;
        foldIdxEdges(features.edgesFtLb(:,1) == sceneIdx(i)) = mod(i, numFolds) + 1;
      end
      
      folds = cell(1,numFolds);
      structuredFactors = cell(numFolds, 1);
      spatialFactors = cell(numFolds, size(self.ftypes,1));
      matlabpool('open', self.num_jobs);
      cleaner = onCleanup(@() matlabpool('close'));
      numLabels = self.nLabels;
      numFtypes = size(self.ftypes, 1);
      foldNodeFtLbs = cell(1,numFolds);
      foldEdgeFtLbs = cell(1,numFolds);
      for foldnum = 1:numFolds
        foldNodeFtLbs{foldnum} = features.nodesFtLb(foldIdxNodes == foldnum, :);
        foldEdgeFtLbs{foldnum} = features.edgesFtLb(foldIdxEdges == foldnum, :);
      end
      for foldnum = 1:numFolds
        foldNodeFtLb = foldNodeFtLbs{foldnum};
        foldEdgeFtLb = foldEdgeFtLbs{foldnum};
        folds{foldnum}.features = foldNodeFtLb(:,4:end);
        %folds{foldnum}.labels = foldNodeFtLb(:,3); %TODO: overcomplete!
        % assume labels are in 1..n
        assert(~isempty(folds{foldnum}.features));
        
        % HACK: later become zero
        tmpLabels = foldNodeFtLb(:,3);
        tmpLabels(foldNodeFtLb(:,3) == 0) = 1;
        folds{foldnum}.labels = zeros(size(foldNodeFtLb, 1), numLabels); 
        folds{foldnum}.labels((1:size(foldNodeFtLb, 1))' + (tmpLabels-1)*size(foldNodeFtLb, 1)) = 1;
        %folds{foldnum}.labels(foldNodeFtLb(:,3) == 0, :) = 0; % TODO: not good when all zeros

        %numSnakes = size(foldSnakes, 1);
        %folds{foldnum}.labels = repmat(eye(10), numSnakes, 1); % overcomplete
        

        % STRUCTURED LINKS
        structuredFactors{foldnum} = cell(1, size(foldEdgeFtLb, 1));
        for i = 1:size(foldEdgeFtLb, 1)
          structuredFactors{foldnum}{i}.ftypenum = 0;
          % TODO: possible to cache for speed
          structuredFactors{foldnum}{i}.mynum = ...
            find(foldNodeFtLb(:,1) == foldEdgeFtLb(i,1) & ...
              foldNodeFtLb(:,2) == foldEdgeFtLb(i,2));
          assert(numel(structuredFactors{foldnum}{i}.mynum) == 1);
          structuredFactors{foldnum}{i}.neighnums = ...
            find(foldNodeFtLb(:,1) == foldEdgeFtLb(i,1) & ...
              foldNodeFtLb(:,2) == foldEdgeFtLb(i,3));
          structuredFactors{foldnum}{i}.pwfeatures = foldEdgeFtLb(i,4:end);
        end
        
        % SPATIAL LINKS
        for ftype = 1:numFtypes
          spatialFactors{foldnum, ftype} = cell(1, size(foldNodeFtLb, 1));
        end
        for node = 1:size(foldNodeFtLb, 1)
          cloudNum = foldNodeFtLb(node,1);
          %cloudSegList = foldNodeFtLb(foldNodeFtLb(:,1) == cloudNum,2);
          cloudSegList = foldNodeFtLb(:,2);
          cloudSegList(foldNodeFtLb(:,1) ~= cloudNum) = -1; % HACK: this will never show up in intersect below
          cloudFile = sprintf(pcdPathMask, cloudNum);
          nodeFact = cell(numFtypes, 1);
          parfor ftype = 1:numFtypes
            nodeFact{ftype}.ftypenum = ftype;
            nodeFact{ftype}.mynum = node;
            
            %tic;
            hst = region_segments(cloudFile, foldNodeFtLb(node,2), self.ftypes(ftype,:), 0, 1); %#ok<PFBNS>
            %toc
            [~, neighIdx, nodeIdx] = intersect(hst(:,1), cloudSegList, 'rows');
            nodeFact{ftype}.neighnums = nodeIdx;
            nodeFact{ftype}.neighweights = double(hst(neighIdx,2));
            nodeFact{ftype}.pwfeatures = [];  % not used
          end
          for ftype = 1:numFtypes
            spatialFactors{foldnum,ftype}{node} = nodeFact{ftype};
          end
        end
        
        fprintf('Fold %d done\n', foldnum);
      end
    end
  end
  
end

