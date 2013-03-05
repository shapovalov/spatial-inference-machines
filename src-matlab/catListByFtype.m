function [features, labels, featRevIdx] = ...
  catListByFtype(foldList, ftypenum, exclFolds, prevLabels, exclIsUse, featConfig)
% This function retains factors of the given type concatenated over folds.
% Some folds may be hold out, as needed during training.
% It returns generalized features of the factors (i.e. including previous iteration labels)
% along with ground truth labels, if available.
%INPUT:
% foldList: cell array of folds. Each fold fold is a struct with fields:
%   labels: matrix of size nvars x nlabels with over-complete label
%     representation (optional)
%   features: matrix of size nvars x nfeatures with local features of variables
%   factors: cell array of d-factors; each element includes:
%     ftypenum: corresponding factor type ID
%     mynum: index of variable within the fold that corresponds to the factor destination
%     neighnums: indices of variables in factor source
%     pwfeatures: array of pairwise features for the factor; default: empty
%     neighweights: weights of the impact of individual source points; default: uniform
% ftypenum: the type ID of factors that should be retained
% exclFolds: (optional) array of folds that are excluded; default: empty
% prevLabels: (optional) cell array of the lenght numfolds containing
%   previous iteration labels for each fold's variables. They are included to 
%   generalized features of d-factors, when available.
%   prevLabels{i} is of the size nvars x nlabels; default: empty
% exclInUse: flag that inverts exclFolds; i.e. exclFolds ARE used, not ignored;
%   default: false
% featConfig: (optional) struct that defines what should be included to
%     generalized features vector; fields:
%   myfeatures: features of the destination variable (default: true)
%   neighfeatures: mean features of the source variables (default: false)
%   mylabels: previous iteration label of the destination variable (default: true)
%   neighlabels: mean previous iteration labels of the source variables (default: true)
%   pwfeatures: features of the d-factor (default: true)
%OUTPUT:
% features: matrix of concatenated generalized features of d-factors (nfactors x ngenfeatures)
% labels: matrix of ground truth labels for factor destinations (nfactors x nlabels).
%   If foldList{1}.labels are not provided, returns empty matrix.
% featRevIdx: indices of the returned factors in foldList (nfactors x 2).
%   featRevIdx(i,1) is the fold index, featRevIdx(i,2) is the index of factor in the fold

  assert(nargin >= 2);

  if nargin == 2  % default: use all folds
    exclFolds = [];
  end

  if nargin < 5
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
  if (nargin >= 4 && ~isempty(prevLabels)) 
    % not the first round; add labels from previous iteration to features
    assert(length(prevLabels) == length(foldList)); 
    assert(isempty(prevLabels{1}) || isempty(foldList{1}.labels) || ...
      size(prevLabels{1}, 1) == size(foldList{1}.labels, 1));
    numOldLabels = size(prevLabels{1}, 2);
  end
  
  featConfigDef = struct('myfeatures', true, 'neighfeatures', false, ...
                         'mylabels', true, 'neighlabels', true, 'pwfeatures', true);
  if exist('featConfig', 'var')  
    % if featConfig is (at least) partially given, use its fields for initialization
    for f = fieldnames(featConfig)'
      featConfigDef.(f{1}) = featConfig.(f{1});
    end
  end
  featConfig = featConfigDef;

  % pre-allocate
  numfactors = sum(cellfun(@(fold) ...
      sum(cellfun(@(factor) factor.ftypenum == ftypenum, fold.factors)), ...
    foldList(foldNums)));

  numfeaturesTemp = size(foldList{1}.features, 2) ... % my  features only
    + numOldLabels*2;
  
  numfeatures = size(foldList{1}.features, 2) * ...
      int32(featConfig.myfeatures + featConfig.neighfeatures) + ...
    numOldLabels * int32(featConfig.mylabels + featConfig.neighlabels);
  % TEMP check refactoring
  assert(numfeaturesTemp == numfeatures);

   
  % find any factor of the ftypenum
  if featConfig.pwfeatures
    fctnum = 0;
    for i = 1:length(foldList{1}.factors)
      if foldList{1}.factors{i}.ftypenum == ftypenum
        fctnum = i;
        break
      end
    end
    if fctnum == 0
      error('catListByFtype: the first fold does not contain d-factors of type %d', ftypenum);
    end
    if isfield(foldList{1}.factors{fctnum}, 'pwfeatures') && ...
        ~isempty(foldList{1}.factors{fctnum}.pwfeatures)
      numfeatures = numfeatures + length(foldList{1}.factors{1}.pwfeatures);
    end
  end
  
  features = zeros(numfactors, numfeatures);


  labels = zeros(numfactors, size(foldList{1}.labels, 2)); % my labels
  featRevIdx = zeros(numfactors, 2);  

  numFact = 0;
  for foldNum = foldNums
    fold = foldList{foldNum};
    for factorNum = 1:length(fold.factors)
      factor = fold.factors{factorNum};
      if factor.ftypenum ~= ftypenum
        continue
      end
      
      factorFeatures = [];
      if featConfig.myfeatures
        factorFeatures = [factorFeatures, fold.features(factor.mynum,:)]; %#ok<AGROW>
      end
      if featConfig.neighfeatures
        factorFeatures = [factorFeatures, mean(fold.features(factor.neighnums,:), 1)]; %#ok<AGROW>
      end
      if featConfig.pwfeatures && isfield(factor, 'pwfeatures') && ...
          ~isempty(factor.pwfeatures)
        factorFeatures = [factorFeatures, factor.pwfeatures]; %#ok<AGROW>
      end
      
      if numOldLabels > 0  % not the first training iteration; add previous iter labels
        if featConfig.neighlabels
          if isfield(factor, 'neighweights') && ~isempty(factor.neighweights)
            weightedNeighLabs = sum(...
              bsxfun(@times, prevLabels{foldNum}(factor.neighnums, :), factor.neighweights), 1) / ...
              sum(factor.neighweights, 1);
            assert(all(~isnan(weightedNeighLabs)));
            assert(abs(sum(weightedNeighLabs) - 1.0) < 1e-6);
          else % equal contribution by neighbours
            weightedNeighLabs = mean(prevLabels{foldNum}(factor.neighnums, :), 1);
          end
          weightedNeighLabs(isnan(weightedNeighLabs)) = 0;
          factorFeatures = [factorFeatures, weightedNeighLabs]; %#ok<AGROW>
        end

        if featConfig.mylabels
          factorFeatures = [factorFeatures, prevLabels{foldNum}(factor.mynum, :)]; %#ok<AGROW>
        end
      end
      assert(all(all((~isnan(factorFeatures)))));

      numFact = numFact + 1;
      features(numFact,:) = factorFeatures;
      featRevIdx(numFact,1) = foldNum;
      featRevIdx(numFact,2) = factor.mynum;
      if ~isempty(labels) % ground truth labels are given
        labels(numFact,:) = fold.labels(factor.mynum,:);
      end
    end
  end
  
  assert(numfactors == numFact); % the same number of factors as with reduction
end
