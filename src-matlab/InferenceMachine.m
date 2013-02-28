classdef InferenceMachine < handle
  %INFERENCEMACHINE Performs training and inference with sequential classification
  
  properties (Constant)
    PROB_EPS = 0.001;
  end
  
  properties (SetAccess = private, GetAccess = private)
    classifiers
    reg_coef = 0
    num_jobs
  end
  
  properties (SetAccess = private)
    isTrained = false
  end
  
  methods
    function self = InferenceMachine(num_jobs)
      if ~exist('num_jobs', 'var')
        num_jobs = 6;
      end
      self.num_jobs = num_jobs;
    end
    
    function testLabels = train(self, samplers) % interface may be changed
      % TEMP debug output
      fprintf('\nStart training, reg coef = %f\n', self.reg_coef);
      
      numRounds = length(samplers);
      self.classifiers = self.initClassifiersArray(numRounds, samplers{1}.getNumFtypes());
      %labels = samplers{1}.getPriorLabels();
      labels = cell(1,samplers{1}.getNumFolds());
      
      testLabels = cell(1,numRounds); % the result, if the samplers require test
      
      matlabpool('open', self.num_jobs);
      cleaner = onCleanup(@() matlabpool('close'));
      
      for round = 1:numRounds
        fprintf('round #%d\n', round);
        prevLabels = labels;
        labels = cell(1, samplers{round}.getNumFolds());
        gtLabelsGlobal = cell(1, samplers{round}.getNumFolds());
        accPrevFact = -Inf;
        
        % first, train temporary classifiers on hold-out examples
        while true
          % TEMP
          %[factorPacks ftypenum] = samplers{round}.sample(accPrevFact, round == numRounds, round);
          %[factorPacks ftypenum] = samplers{round}.sample(accPrevFact, true, round);
          [factorPacks ftypenum] = samplers{round}.sample(accPrevFact);
          if isempty(factorPacks)
            break
          end
          
          %TEMP
          disp(ftypenum);
          
          for foldNum = 1:samplers{round}.getNumFolds()
            gtLabelsGlobal{foldNum} = factorPacks{foldNum}.labels;
          end
          
          %foldAccs = zeros(samplers{round}.getNumFolds(), 2); % sums and nums
          %labelAccs = zeros(size(factorPacks{1}.labels,2), 2); % sums and nums
          gtLabDists = zeros(size(factorPacks{1}.labels,2), size(factorPacks{1}.labels,2)); % for entropy
          parfor foldNum = 1:samplers{round}.getNumFolds()
          %for foldNum = 1:samplers{round}.getNumFolds()
            tmpClassifier = InferenceMachine.initClassifiersArray(1, 1); 
            tmpClassifier = tmpClassifier{1};
          
            [roundFeat, roundLab] = catListByFtype(factorPacks, ftypenum, foldNum, prevLabels);
            tmpClassifier.train(roundFeat, roundLab); 
            assert(tmpClassifier.isTrained);

            [testFeat, gtLabels, featRevIdx] = catListByFtype(factorPacks, ftypenum, foldNum, prevLabels, true);
            assert(all(featRevIdx(:,1) == foldNum)); % all sampled from this fold
            ftypeLabels = tmpClassifier.classify(testFeat);
            assert(size(ftypeLabels,1) == length(featRevIdx));
            % now transform to labels for folds
            if isempty(labels{foldNum})
              labels{foldNum} = cell(1, size(factorPacks{foldNum}.labels, 1));
            end
            
            % HACK: regularization to cope with biased classifiers
            ftypeLabels = ftypeLabels .* (1 - 2*InferenceMachine.PROB_EPS) + InferenceMachine.PROB_EPS;
            
            for siteNum = 1:length(featRevIdx)
              assert(featRevIdx(siteNum,1) == foldNum); % all sampled from this fold
              labels{foldNum}{featRevIdx(siteNum,2)} = ...
                [labels{foldNum}{featRevIdx(siteNum,2)}; ftypeLabels(siteNum,:) ftypenum];
            end
            
            % likelihood as acc measure. 
%             labLkhood = ftypeLabels(gtLabels > 0.5);
%             assert(length(labLkhood) == size(gtLabels, 1));
%             assert(~isempty(labLkhood));
%             %foldAccs(foldNum,1) = sum(log(labLkhood));
%             foldAccs(foldNum,1) = mean(log(labLkhood));
            
            %labelAccs = labelAccs + [sum(log(ftypeLabels) .* (gtLabels > 0.5))',...
            %  sum(gtLabels > 0.5)'];
            
            % entropy
            gtLabDists = gtLabDists + ftypeLabels' * (gtLabels > 0.5);
          end
%           foldAccs = sum(foldAccs);
%           % normalize to penalize sparce features; here we implicitely assume that
%           % label could be only associated with a single factor of certain type 
%           %acc = [acc (foldAccs(2)-foldAccs(1))/sum(cellfun(@(fp) size(fp.labels,1), factorPacks))];
%           %acc = [acc (foldAccs(2)-foldAccs(1))/foldAccs(2)];
%           accPrevFact = (foldAccs(2)-foldAccs(1))/foldAccs(2);
          %accPrevFact = sum(foldAccs(:,1));
          %accPrevFact = mean(labelAccs(:,1) ./ labelAccs(:,2));

          %accPrevFact = max(labelAccs(:,1) ./ labelAccs(:,2));
          
          % minus entropy
          gtLabDists = bsxfun(@rdivide, gtLabDists, sum(gtLabDists, 1)); % balance gt labels
          ftsum = sum(gtLabDists, 2); ftsum = ftsum / sum(ftsum);
          gtLabDists = bsxfun(@rdivide, gtLabDists, sum(gtLabDists, 2));
          labEntropys = sum(gtLabDists .* log(gtLabDists), 2);
          accPrevFact = sum(ftsum .* labEntropys); 
          %disp(gtLabDists); disp(labEntropys); disp(ftsum); 
          disp(accPrevFact);
        end
        % normalize labels; we did not count the number of voting factors
        % explicitely. Here we take the mean instead of the product!
        %labels = cellfun(@(pkLab) bsxfun(@rdivide, pkLab, sum(pkLab, 2)), labels, 'UniformOutput', false);
        
        idx = samplers{round}.getActiveFtypes();
        if self.reg_coef < 10
          %curriedLkhood = @(alpha) getFtypeCoefLkhood(alpha, labels, gtLabelsGlobal, self.reg_coef);
          curriedLkhood = @(alpha) getFtypeCoefRegr(alpha, labels, gtLabelsGlobal, self.reg_coef);
          ftypePowers = fminunc(curriedLkhood, zeros(length(idx),1), ... %zeros -- exp
            optimset('GradObj','on','Display','iter-detailed'));
          %ftypePowers = fminunc(curriedLkhood, ones(length(idx),1), ... %zeros -- exp
            %optimset('GradObj','on', 'Hessian','on','Display','final-detailed'));
        else
          ftypePowers = zeros(length(idx),1); % no tuning
        end
        ftypePowers = exp(ftypePowers); % since we in fact optimize log(powers)
        disp(ftypePowers);
        %ftypePowers = ones(length(idx),1);  % no powers!
        samplers{round}.setFtypePowers(ftypePowers);
        
        % TEMP save votes
        %if round == numRounds
         %global gl_rad
         %save(sprintf('seabattle/fctlabels-FANAL-r%d-seas4dk-10mcmc30-linneigh-meanlogaccClEnt-resamp-reg-featEps005.mat', ...
         %              round), 'labels');
        %end
        
%         labels = cellfun(@(pkLab) ...
%             cell2mat(cellfun(@(labLabs) ...
%                  prod(labLabs(any(bsxfun(@eq, labLabs(:,end), idx), 2), 1:end-1), 1), ... % change prod/mean
%               pkLab, 'UniformOutput', false)'), ... 
%           labels, 'UniformOutput', false);
%         
        nLabels = size(labels{1}{1}, 2) - 1;
        for i = 1:length(labels)
          newlab = ones(length(labels{i}), nLabels) / nLabels; %TEMP
          for j = 1:length(labels{i})
            if isempty(labels{i}{j})
              if ~isempty(prevLabels{i})
                newlab(j,:) = prevLabels{i}(j,:);
              end
            else  
              %newlab(j,:) = prod(labels{i}{j}(any(bsxfun(@eq, labels{i}{j}(:,end), idx), 2), 1:end-1), 1);
              %newlab(j,:) = 1;
              newlab(j,:) = 0;
              for fct = 1:size(labels{i}{j},1)
                ftypeActIdx = find(idx == labels{i}{j}(fct,end));
                if isempty(ftypeActIdx)
                  continue
                end
                assert(numel(ftypeActIdx) == 1);
                %newlab(j,:) = newlab(j,:) .* labels{i}{j}(fct,1:end-1).^ftypePowers(ftypeActIdx);
                newlab(j,:) = newlab(j,:) + labels{i}{j}(fct,1:end-1).*ftypePowers(ftypeActIdx);
              end
            end
          end
          labels{i} = newlab;
          %labels{i} = exp(newlab);
        end
        
        % normalize prods
        labels = cellfun(@(pkLab) ...
            bsxfun(@rdivide, pkLab, sum(pkLab,2)), ... 
          labels, 'UniformOutput', false);
        % HACK: set all 0s for labels w/o factors (mean returns NaNs)
        %for i = 1:length(labels),  labels{i}(isnan(labels{i})) = 0; end
        for i = 1:length(labels),  labels{i}(isnan(labels{i})) = 1/nLabels; end
        assert(all(all(~isnan(labels{1}))));  % check only the first fold
        
        factorPacks = samplers{round}.sampleActiveFactors();
        
        % then, train final classifier 
        for ftypenumnum = 1:length(idx)
          ftypenum = idx(ftypenumnum);
          [roundFeat, roundLab] = catListByFtype(factorPacks, ftypenum, [], prevLabels);
          self.classifiers{round, ftypenumnum}.train(roundFeat, roundLab, 0);
        end
        
        %if nargin > 2
        testFactorPacks = samplers{round}.sampleActiveFactors(true); %HACK!
        if ~isempty(testFactorPacks)
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
          %labels = samplers{1}.getPriorTestLabels();
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
              %labels{i} = zeros(size(factorPacks{i}.labels, 1), size(factorPacks{i}.labels, 2)); % mean
              labels{i} = ones(size(factorPacks{i}.labels, 1), size(factorPacks{i}.labels, 2)); % prod
              
              %TEMP
              %labels{i}(1:10:end,1) = 1;
            end
          end
          
          % HACK: regularization to cope with biased classifiers
          ftypeLabels = ftypeLabels .* (1 - 2*self.PROB_EPS) + self.PROB_EPS;
            
          ftypePowers = samplers{round}.getFtypePowers();
          for siteNum = 1:length(featRevIdx)
            labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :) = ...
                            labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :) + ...
                             ftypeLabels(siteNum,:).*ftypePowers(ftypenumnum); %log regr
              %bsxfun(@times, labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :), ...
              %               ftypeLabels(siteNum,:).^ftypePowers(ftypenumnum)); %prod w/ powers
              
              %bsxfun(@times, labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :), ...
              %               ftypeLabels(siteNum,:).^ftypePowers(ftypenumnum)); %prod w/ powers
              %bsxfun(@times, labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :), ftypeLabels(siteNum,:)); %prod
              %labels{featRevIdx(siteNum,1)}(featRevIdx(siteNum,2), :) + ftypeLabels(siteNum,:); % mean

          end
        end
        labels = cellfun(@(pkLab) exp(pkLab), labels, 'UniformOutput', false); % log regr
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
      for i = 1:size(classifiers, 1)
        for j = 1:size(classifiers, 2)
          %classifiers{i,j} = LogitRegression();
          classifiers{i,j} = RandomForest();
        end
      end
    end
    
    function [res] = softmax(alpha, x)
      expax = exp(x*alpha);
      res = sum(x .* expax) / sum(expax);
    end
  end
  
end

function jointLabs = inferLabsProd(labLabs, prevLabs)
      jointLabs = prod(labLabs(any(bsxfun(@eq, labLabs(:,end), idx), 2), 1:end-1), 1);
end
    
