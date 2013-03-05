classdef InferenceMachine < handle
  %INFERENCEMACHINE Performs training and inference with sequential classification
  
  properties (Constant)
    PROB_EPS = 0.001;
  end
  
  properties (SetAccess = private, GetAccess = private)
    classifiers
    reg_coef = 0 % pass anything < 0 to use uniform factor type weights
    num_jobs
    lkhood_ftype_powers
  end
  
  properties (SetAccess = private)
    isTrained = false
  end
  
  methods
    function self = InferenceMachine(num_jobs, lkhood_ftype_powers)
      if ~exist('num_jobs', 'var')
        num_jobs = 6;
      end
      self.num_jobs = num_jobs;
      
      if ~exist('lkhood_ftype_powers', 'var')
        lkhood_ftype_powers = true;
      end
      self.lkhood_ftype_powers = lkhood_ftype_powers;
    end
    
    function testLabels = train(self, samplers) 
      numRounds = length(samplers);
      self.classifiers = self.initClassifiersArray(numRounds, samplers{1}.getNumFtypes());
      labels = cell(1,samplers{1}.getNumFolds());
      
      testLabels = cell(1,numRounds); % the result, if the samplers require test
      
      matlabpool('open', self.num_jobs);
      cleaner = onCleanup(@() matlabpool('close'));
      
      for round = 1:numRounds
        fprintf('Iteration #%d\n', round);
        prevLabels = labels;
        labels = cell(1, samplers{round}.getNumFolds());
        gtLabelsGlobal = cell(1, samplers{round}.getNumFolds());
        accPrevFact = -Inf;
        
        % first, train temporary classifiers on hold-out examples
        while true   % loop over factor types
          [factorPacks ftypenum] = samplers{round}.sample(accPrevFact);
          if isempty(factorPacks) % iteration is finished
            break
          end
          
          for foldNum = 1:samplers{round}.getNumFolds()
            gtLabelsGlobal{foldNum} = factorPacks{foldNum}.labels;
          end
          
          gtLabDists = zeros(size(factorPacks{1}.labels,2), size(factorPacks{1}.labels,2)); % for entropy
          parfor foldNum = 1:samplers{round}.getNumFolds()
            tmpClassifier = InferenceMachine.initClassifiersArray(1, 1); 
            tmpClassifier = tmpClassifier{1};
          
            [roundFeat, roundLab] = catListByFtype(factorPacks, ftypenum, foldNum, prevLabels);
            tmpClassifier.train(roundFeat, roundLab); 
            assert(tmpClassifier.isTrained);

            % classify the hold-out examples
            [testFeat, gtLabels, featRevIdx] = catListByFtype(factorPacks, ftypenum, foldNum, prevLabels, true);
            ftypeLabels = tmpClassifier.classify(testFeat);
            assert(size(ftypeLabels,1) == length(featRevIdx));
            % now transform to labels for folds
            if isempty(labels{foldNum})
              labels{foldNum} = cell(1, size(factorPacks{foldNum}.labels, 1));
            end
            
            % HACK: regularization to cope with biased classifiers
            ftypeLabels = ftypeLabels .* (1 - 2*InferenceMachine.PROB_EPS) + InferenceMachine.PROB_EPS;
            
            % aggregate labels by destination variables
            for siteNum = 1:length(featRevIdx)
              assert(featRevIdx(siteNum,1) == foldNum); % all sampled from this fold
              labels{foldNum}{featRevIdx(siteNum,2)} = ...
                [labels{foldNum}{featRevIdx(siteNum,2)}; ftypeLabels(siteNum,:) ftypenum];
            end
            
            % to compute entropy (relevant for non-static factor samplers)
            gtLabDists = gtLabDists + ftypeLabels' * (gtLabels > 0.5);
          end  % loop over factor types
          
          % minus entropy
          gtLabDists = bsxfun(@rdivide, gtLabDists, sum(gtLabDists, 1)); % balance gt labels
          ftsum = sum(gtLabDists, 2); ftsum = ftsum / sum(ftsum);
          gtLabDists = bsxfun(@rdivide, gtLabDists, sum(gtLabDists, 2));
          labEntropys = sum(gtLabDists .* log(gtLabDists), 2);
          accPrevFact = sum(ftsum .* labEntropys); 
        end  % loop over folds

        % find optimal factor type weights (powers)
        idx = samplers{round}.getActiveFtypes();
        if self.reg_coef >= 0 
          % the following function optimized log(powers) to ensure powers > 0
          if self.lkhood_ftype_powers
            curriedLkhood = @(alpha) getFtypeCoefLkhood(alpha, labels, gtLabelsGlobal, self.reg_coef);
          else
            curriedLkhood = @(alpha) getFtypeCoefRegr(alpha, labels, gtLabelsGlobal, self.reg_coef);
          end
          ftypePowers = fminunc(curriedLkhood, zeros(length(idx),1), ... %zeros -- exp
            optimset('GradObj','on','Display','final-detailed'));
        else  % if self.reg_coef < 0, set uniform weights
          ftypePowers = zeros(length(idx),1); % no tuning
        end
        ftypePowers = exp(ftypePowers); % since we in fact optimize log(powers)
        disp(ftypePowers);
        samplers{round}.setFtypePowers(ftypePowers);
            
        
        nLabels = size(labels{1}{1}, 2) - 1;
        for fold = 1:length(labels)
          % default for nodes that are not destinations. Should not be often used.
          newlab = ones(length(labels{fold}), nLabels) / nLabels;
          for node = 1:length(labels{fold})
            if isempty(labels{fold}{node})  
              if ~isempty(prevLabels{fold}) % if topology changes over rounds
                newlab(node,:) = prevLabels{fold}(node,:);
              end
            else  
              newlab(node,:) = double(self.lkhood_ftype_powers);
              for fct = 1:size(labels{fold}{node},1)
                % find if the factor type is active, and if yes, find its index
                ftypeActIdx = find(idx == labels{fold}{node}(fct,end));
                if isempty(ftypeActIdx)
                  continue
                end
                assert(numel(ftypeActIdx) == 1);
                if self.lkhood_ftype_powers
                  newlab(node,:) = newlab(node,:) .* labels{fold}{node}(fct,1:end-1).^ftypePowers(ftypeActIdx);
                else
                  newlab(node,:) = newlab(node,:) + log(labels{fold}{node}(fct,1:end-1)).*ftypePowers(ftypeActIdx);
                end
              end
            end
          end
          if self.lkhood_ftype_powers
            labels{fold} = newlab;
          else
            labels{fold} = exp(newlab);
          end
        end
        
        % normalize prods
        labels = cellfun(@(pkLab) ...
            bsxfun(@rdivide, pkLab, sum(pkLab,2)), ... 
          labels, 'UniformOutput', false);
        assert(all(all(~isnan(labels{1}))));  % check only the first fold
        
        factorPacks = samplers{round}.sampleActiveFactors();
        
        % then, train final classifier 
        for ftypenumnum = 1:length(idx)
          ftypenum = idx(ftypenumnum);
          [roundFeat, roundLab] = catListByFtype(factorPacks, ftypenum, [], prevLabels);
          self.classifiers{round, ftypenumnum}.train(roundFeat, roundLab, 0);
        end
        
        % test, if needed
        if samplers{round}.testDataAreSet()
          if round == 1
            testLabels{round} = self.inferImpl(samplers, round);
          else
            testLabels{round} = self.inferImpl(samplers, round, testLabels{round-1});
          end
        end
      end
      
      self.isTrained = true;
    end
    
    
    
    function labels = infer(self, samplers)
      assert(self.isTrained);
      
      labels = self.inferImpl(samplers, 1:length(samplers));
    end
    
    function setRegCoef(self, coef)
      self.reg_coef = coef;
    end
  end
  
  
  
  methods (Access = private)
    function labels = inferImpl(self, samplers, roundNums, labels)
      assert(all(sort(roundNums) == roundNums)); % roundNums are sorted
      assert(nargin >= 4 || roundNums(1) == 1);  % either labels are given or start from first round
         
      for round = roundNums
        [factorPacks idx] = samplers{round}.sampleActiveFactors(true);
        assert(~isempty(factorPacks));
        
        if ~exist('labels', 'var') || isempty(labels)
          labels = cell(1,1);
        end
        prevLabels = labels;
        labels = cell(1, length(factorPacks));
        
        for ftypenumnum = 1:length(idx)
          ftypenum = idx(ftypenumnum);
          [roundFeat, ~, featRevIdx] = catListByFtype(factorPacks, ftypenum, [], prevLabels);

          ftypeLabels = self.classifiers{round, ftypenumnum}.classify(roundFeat);
          assert(size(ftypeLabels,1) == length(featRevIdx));
          % now transform to labels for folds
          if isempty(labels{1})
            for i = 1:length(labels)
              if self.lkhood_ftype_powers
                labels{i} = ones(size(factorPacks{i}.labels, 1), size(factorPacks{i}.labels, 2)); % prod
              else
                labels{i} = zeros(size(factorPacks{i}.labels, 1), size(factorPacks{i}.labels, 2)); % mean, log regr
              end
            end
          end
          
          % HACK: regularization to cope with biased classifiers
          ftypeLabels = ftypeLabels .* (1 - 2*self.PROB_EPS) + self.PROB_EPS;
            
          ftypePowers = samplers{round}.getFtypePowers();
          for siteNum = 1:length(featRevIdx)
            if self.lkhood_ftype_powers
              labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :) = ...
                bsxfun(@times, labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :), ...
                               ftypeLabels(siteNum,:).^ftypePowers(ftypenumnum)); %prod w/ powers
            else
              labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :) = ...
                labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :) + ...
                  log(ftypeLabels(siteNum,:)).*ftypePowers(ftypenumnum); %log regr
            end
          end
        end
        if ~self.lkhood_ftype_powers
          labels = cellfun(@(pkLab) exp(pkLab), labels, 'UniformOutput', false); % log regr
        end
        % normalize labels; we did not count the number of voting factors
        % explicitely. Here we take the mean instead of the product!
        labels = cellfun(@(pkLab) bsxfun(@rdivide, pkLab, sum(pkLab, 2)), labels, 'UniformOutput', false);
        
        % HACK: set all 0s for labels w/o factors (mean returns NaNs)
        for i = 1:length(labels),  labels{i}(isnan(labels{i})) = 1/size(labels{1}, 2); end
        assert(all(all(~isnan(labels{1}))));  % check only the first fold
      end
    end
  end
  
  
  methods (Static = true, Access = private)
    function classifiers = initClassifiersArray(y, x)
      classifiers = cell(y, x); 
      classifiers = cellfun(@(~)RandomForest(), classifiers, 'UniformOutput', false);
    end
    
    function [res] = softmax(alpha, x)
      expax = exp(x*alpha);
      res = sum(x .* expax) / sum(expax);
    end
  end
  
end
    
