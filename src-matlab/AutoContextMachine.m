classdef AutoContextMachine < handle
  %AutoContextMachine Performs training and inference with sequential classification based on the paper of Tu [2008], but with hold-out estimation of with previous-iteration labels
  
  properties (SetAccess = private, GetAccess = private)
    classifiers
    num_jobs
  end
  
  properties (SetAccess = private)
    isTrained = false
    meanDepth
  end
  
  methods
    % Constructs the class. Input arguments:
    % num_jobs: number of parallel training jobs (default: 6)
    function self = AutoContextMachine(num_jobs)
      if ~exist('num_jobs', 'var')
        num_jobs = 6;
      end
      self.num_jobs = num_jobs;
    end
    
    % Trains auto-context using the set encapsulated in factor
    % samplers. If a test set is additionally provided, infers its labels.
    %INPUT:
    % samplers: cell array of the length equal to the number of rounds.
    %   Elements are instantiations of FactorSampler that store training
    %   set and (optionally) test set
    %OUTPUT:
    % trainLabels: cell array of length nRounds, each element is cell array
    %   of inferred training set labels divided by folds
    % testLabels: if samplers{round}.testDataAreSet(), then
    %   testLabels{round} contains an over-complete representations for
    %   inferred labels for the test set
    function [trainLabels testLabels] = train(self, samplers) 
      numRounds = length(samplers);
      self.classifiers = self.initClassifiersArray(numRounds, 1);
      labels = cell(1,samplers{1}.getNumFolds());
      
      if nargout > 0
        trainLabels = cell(1,numRounds);
      end
      if nargout > 1
        testLabels = cell(1,numRounds); % the result, if the samplers require test
      end
      
      matlabpool('open', self.num_jobs);
      cleaner = onCleanup(@() matlabpool('close'));
      
      for round = 1:numRounds
        fprintf('Iteration #%d\n', round);
        prevLabels = labels;
        labels = cell(1, samplers{round}.getNumFolds());
        
        % first, train temporary classifiers on hold-out examples
        [factorsPacks actFtypenums] = samplers{round}.sampleActiveFactors(false);
        %assert(all(actFtypenums == 1:length(actFtypenums)));
        
        % pre-compute and swap features for the final classifier
        [roundFeat, roundLab] = self.catListFold(factorsPacks, [], prevLabels); %#ok<NASGU,ASGLU>
        save('featurecache', 'roundFeat', 'roundLab');
        roundFeat = []; %#ok<NASGU>
        roundLab = []; %#ok<NASGU>
        
        if samplers{round}.getNumFolds() > 1  % perform hold-out estimation
          fprintf('Training temporary classifiers\n');

          % precompute features sequentially so we don't need to duplicate factorsPacks
          tic
          roundFeat = cell(1, samplers{round}.getNumFolds());
          roundLab = cell(1, samplers{round}.getNumFolds());
          testFeat = cell(1, samplers{round}.getNumFolds());
          for foldNum = 1:samplers{round}.getNumFolds()
              [roundFeat{foldNum}, roundLab{foldNum}] = AutoContextMachine.catListFold(factorsPacks, foldNum, prevLabels);
              [testFeat{foldNum}, gtLabels, featRevIdx] = AutoContextMachine.catListFold(factorsPacks, foldNum, prevLabels, true);
          end
          fprintf('Time for transforming features: %f\n', toc);

          whos
          factorsPacks = []; %#ok<NASGU>
          whos

          tic
          parfor foldNum = 1:samplers{round}.getNumFolds()
          %for foldNum = 1:samplers{round}.getNumFolds()
            %whos
            tmpClassifier = AutoContextMachine.initClassifiersArray(1, 1); 
            tmpClassifier = tmpClassifier{1};


            disp(size(roundFeat{foldNum}));  % TEMP
            tmpClassifier.train(roundFeat{foldNum}, roundLab{foldNum}); 
            assert(tmpClassifier.isTrained);

            % classify the hold-out examples
            foldLabels = tmpClassifier.classify(testFeat{foldNum});
            %assert(size(foldLabels,1) == length(featRevIdx));

            % now transform to labels for folds
            if isempty(labels{foldNum})
              labels{foldNum} = cell(1, size(roundLab{foldNum}, 1));
            end

            % aggregate labels by destination variables
            %assert(all(featRevIdx(:,1) == foldNum)); % all sampled from this fold
            %assert(all(featRevIdx(:,2) == (1:size(featRevIdx, 1))')); 
            labels{foldNum} = foldLabels;
          end  % loop over folds
          fprintf('Time for training: %f\n', toc);

          roundFeat = [];  %#ok<NASGU> % clear memory
          roundLab = []; %#ok<NASGU>
          testFeat = []; %#ok<NASGU> % clear memory
        end
        
        % then, train final classifier 
        fprintf('Training final classifiers\n');
        %self.meanDepth = zeros(1,length(idx));
        load('featurecache', 'roundFeat', 'roundLab');
        self.classifiers{round}.train(roundFeat, roundLab);
        
        if samplers{round}.getNumFolds() == 1  % NOT perform hold-out estimation
          labels{1} = self.classifiers{round}.classify(roundFeat);
        end
        
        assert(all(all(~isnan(labels{1}))));  % check only the first fold
        if nargout > 0
          trainLabels{round} = labels;
        end
        
        fprintf('Inferring test set labelling\n');
        % test, if needed
        if nargout > 1 && samplers{round}.testDataAreSet()
          if round == 1
            testLabels{round} = self.inferImpl(samplers, round);
          else
            testLabels{round} = self.inferImpl(samplers, round, testLabels{round-1});
          end
        end
      end
      
      self.isTrained = true;
      
      for round = 1:numRounds
        testLabels{round} = {testLabels{round}};
      end
    end
    
    
    
    function [labels allLabels] = infer(self, samplers)
      assert(self.isTrained);
      
      if nargin < 2
        labels = self.inferImpl(samplers, 1:length(samplers));
      else
        [labels allLabels] = self.inferImpl(samplers, 1:length(samplers));
      end
    end
  end
  
  
  
  methods (Access = private)
    function [labels allLabels] = inferImpl(self, samplers, roundNums, labels)
      assert(all(sort(roundNums) == roundNums)); % roundNums are sorted
      assert(nargin >= 4 || roundNums(1) == 1);  % either labels are given or start from first round
         
      if nargin > 1
        allLabels = cell(1,max(roundNums));
      end
      
      for round = roundNums
        [factorPacks idx] = samplers{round}.sampleActiveFactors(true);
        assert(~isempty(factorPacks));
        %assert(all(idx == 1 : length(idx)));
        
        if ~exist('labels', 'var') || isempty(labels)
          labels = [];
        end
        prevLabels = {labels};

        [roundFeat, ~, featRevIdx] = self.catListFold(factorPacks, [], prevLabels);

        labels = self.classifiers{round}.classify(roundFeat);
        assert(size(labels,1) == length(featRevIdx));

        if nargin > 1
          allLabels{round} = labels;
        end
      end
    end
    
  end
  
  
  methods (Static = true, Access = private)
  
    function [features, labels, featRevIdx] = ...
          catListFold(foldList, exclFolds, prevLabels, exclIsUse, featConfig)
      assert(nargin >= 1);

      if nargin == 1  % default: use all folds
        exclFolds = [];
      end

      if nargin < 4
        exclIsUse = false;
      end

      if exclIsUse
        foldNums = exclFolds;
      else
        foldNums = setdiff(1:length(foldList), exclFolds);
      end

      assert(all(foldNums <= length(foldList)));

      if ~isfield(foldList{1}, 'labels')
        foldList{1}.labels = []; % if labels are not given, mark as empty matrix
      end

      numOldLabels = 0;
      if (nargin >= 3 && ~isempty(prevLabels)) 
        % not the first round; add labels from previous iteration to features
        assert(length(prevLabels) == length(foldList)); 
        assert(isempty(prevLabels{1}) || isempty(foldList{1}.labels) || ...
          size(prevLabels{1}, 1) == size(foldList{1}.labels, 1));
        numOldLabels = size(prevLabels{1}, 2);
      end

      featConfigDef = struct('myfeatures', true, 'neighfeatures', false, ...
                 'mylabels', true, 'neighlabels', true, 'pwfeatures', false);
      if exist('featConfig', 'var')  
        % if featConfig is (at least) partially given, use its fields for initialization
        for f = fieldnames(featConfig)'
          featConfigDef.(f{1}) = featConfig.(f{1});
        end
      end
      featConfig = featConfigDef;

      if featConfig.pwfeatures
        error('Pairwise features are not supported in auto-context');
      end

      maxftype = double(max(cellfun(@(fold) ...
            max(cellfun(@(factor) max(factor.ftypenum), fold.factors)), ...
          foldList(foldNums))));
      minftype = double(min(cellfun(@(fold) ...
            min(cellfun(@(factor) min(factor.ftypenum), fold.factors)), ...
          foldList(foldNums))));
        
      numftypes = maxftype - minftype + 1;

      numfactors = sum(cellfun(  @(fold) size(fold.labels, 1),  foldList(foldNums) ));

      numunfeatures = size(foldList{1}.features, 2);
      numfeatures = numunfeatures * ...
          (int32(featConfig.myfeatures) + numftypes*int32(featConfig.neighfeatures)) + ...
        numOldLabels *  ...
          (int32(featConfig.mylabels) + numftypes*int32(featConfig.neighlabels));
      % structure of the generalized feature vector:
      % [myfeatures, neighfeatures, mylabels, neighlabels]

      % preallocate
      features = zeros(numfactors, numfeatures, 'single');
      labels = zeros(numfactors, size(foldList{1}.labels, 2)); % my labels
      featRevIdx = zeros(numfactors, 2);  

      numFact = 0;
      for foldNum = foldNums
        fold = foldList{foldNum};

        labels(numFact+1 : numFact+size(fold.labels, 1), :) = fold.labels;
        featRevIdx(numFact+1 : numFact+size(fold.labels, 1),1) = foldNum;
        featRevIdx(numFact+1 : numFact+size(fold.labels, 1),2) = 1:size(fold.labels, 1);

        if featConfig.myfeatures
          features(numFact+1 : numFact+size(fold.labels, 1), 1:numunfeatures) = ...
            fold.features;
        end

        if featConfig.mylabels && numOldLabels > 0
          mylabel_start = 0;
          if featConfig.myfeatures,   mylabel_start = mylabel_start + numunfeatures; end
          if featConfig.neighfeatures,   ...
             mylabel_start = mylabel_start + numftypes*numunfeatures; end

          features(numFact+1 : numFact+size(fold.labels, 1), ...
          mylabel_start+1:mylabel_start+numOldLabels) = prevLabels{foldNum};
        end

        for factorNum = 1:length(fold.factors)
          factor = fold.factors{factorNum};
          if featConfig.neighfeatures
            nf_start = int32(featConfig.myfeatures) * numunfeatures + ...
              (factor.ftypenum - minftype) * numunfeatures;  % assume continuous

            features(numFact+factor.mynum, nf_start+1 : nf_start+numunfeatures) = ...
              AutoContextMachine.defmean(fold.features(factor.neighnums,:));
          end

          if featConfig.neighlabels && numOldLabels > 0
            nl_start = 0;
            if featConfig.myfeatures,  nl_start = nl_start + numunfeatures; end
            if featConfig.neighfeatures,   nl_start = nl_start + numftypes*numunfeatures; end
            if featConfig.mylabels,  nl_start = nl_start + numOldLabels; end

            nl_start = nl_start + (factor.ftypenum - minftype) * numOldLabels;  % assume continuous

            features(numFact+factor.mynum, nl_start+1 : nl_start+numOldLabels) = ...
              AutoContextMachine.defmean(prevLabels{foldNum}(factor.neighnums,:));
            % TODO: weighting of old labels!
          end
        end

        numFact = numFact + size(fold.labels, 1);
      end
    end
    
    % mean that defaults to zero
    function res = defmean(matrix)
      if isempty(matrix)
        res = zeros(1, size(matrix,2));
      else
        res = mean(matrix, 1);
      end
    end

    function classifiers = initClassifiersArray(y, x)
      classifiers = cell(y, x); 
      classifiers = cellfun(@(~)RandomForest(), classifiers, 'UniformOutput', false);
      %classifiers = cellfun(@(~)LogitRegression(), classifiers, 'UniformOutput', false);
    end
  end
  
end
    
