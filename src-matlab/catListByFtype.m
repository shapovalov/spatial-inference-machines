function [features, labels, featRevIdx] = ...
  catListByFtype(foldList, ftypenum, exclFolds, prevLabels, exclIsUse)
% This function retains factors of the given type concatenated over folds.
% Some folds may be hold out, as needed during training.
% It returns generalized features of the factors (i.e. including previous iteration labels)
% along with ground truth labels, if available.
%INPUT:
% foldList: cell array of folds. Each fold fold is a struct with fields:
%   labels: matrix of size nvars x nlabels with over-complete label representation
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
%OUTPUT:
% features: matrix of concatenated generalized features of d-factors (nfactors x ngenfeatures)
% labels: matrix of ground truth labels for factor destinations (nfactors x nlabels).
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

  numOldLabels = 0;
  if (nargin >= 4 && ~isempty(prevLabels)) % not the first round; add labels from previous iteration to features
    assert(length(prevLabels) == length(foldList)); % TODO: more checks?
    assert(isempty(prevLabels{1}) || ...
      size(prevLabels{1}, 1) == size(foldList{1}.labels, 1));
    numOldLabels = size(prevLabels{1}, 2);
  end

  % pre-allocate
  numfactors = sum(cellfun(@(fold) ...
    sum(cellfun(@(factor) factor.ftypenum == ftypenum, fold.factors)), ...
    foldList(foldNums)));

  %numfeatures = size(foldList{1}.features, 2)*2 ... % my and mean neighbour's features
  numfeatures = size(foldList{1}.features, 2) ... % my  features only
    + numOldLabels*2;
  %+ numOldLabels; % mean neighbour's labels

  % find any factor of the ftypenum
  fctnum = 0;
  for i = 1:length(foldList{1}.factors)
    if foldList{1}.factors{i}.ftypenum == ftypenum
      fctnum = i;
      break
    end
  end
  assert(fctnum > 0);  % there is at least one factor of this typein the 1st fold
  if isfield(foldList{1}.factors{fctnum}, 'pwfeatures') && ...
      ~isempty(foldList{1}.factors{fctnum}.pwfeatures)
    numfeatures = numfeatures + length(foldList{1}.factors{1}.pwfeatures);
  end
  features = zeros(numfactors, numfeatures);


  labels = zeros(numfactors, size(foldList{1}.labels, 2)); % my labels
  featRevIdx = zeros(numfactors, 2);  % fold num and label num for the current label

  % TODO: rewrite as reduction too!
  numFact = 0;
  for foldNum = foldNums
    fold = foldList{foldNum};
    for factorNum = 1:length(fold.factors)
      factor = fold.factors{factorNum};
      if factor.ftypenum ~= ftypenum
        continue
      end

      factorFeatures = [fold.features(factor.mynum,:), ...
        ]; % no mean neigh features
      %mean(fold.features(factor.neighnums,:), 1)]; % averaging features. is it okay?

      if isfield(factor, 'pwfeatures') && ~isempty(factor.pwfeatures)
        factorFeatures = [factorFeatures, factor.pwfeatures]; %#ok<AGROW>
      end
      if numOldLabels > 0
        if isfield(factor, 'neighweights') && ~isempty(factor.neighweights)
          weightedNeighLabs = sum(...
            bsxfun(@times, prevLabels{foldNum}(factor.neighnums, :), factor.neighweights), 1) / ...
            sum(factor.neighweights, 1);
          assert(all(~isnan(weightedNeighLabs)));
          assert(abs(sum(weightedNeighLabs) - 1.0) < 1e-6);
        else % equal contrib by neighbours
          weightedNeighLabs = mean(prevLabels{foldNum}(factor.neighnums, :), 1);
        end
        weightedNeighLabs(isnan(weightedNeighLabs)) = 0; % for empty factor.neighnums
        %factorFeatures = [factorFeatures, weightedNeighLabs]; %#ok<AGROW>
        % TEMP include my labels
        factorFeatures = [factorFeatures, weightedNeighLabs, ...
          prevLabels{foldNum}(factor.mynum, :)]; %#ok<AGROW>
      end
      factorLabels = fold.labels(factor.mynum,:);
      if ~all(all((~isnan(factorFeatures))))
        save('tmpFactFeatures.mat', 'factorFeatures');
      end
      assert(all(all((~isnan(factorFeatures)))));

      numFact = numFact + 1;
      features(numFact,:) = factorFeatures;
      labels(numFact,:) = factorLabels;
      featRevIdx(numFact,1) = foldNum;
      featRevIdx(numFact,2) = factor.mynum;
    end
  end
  assert(numfactors == numFact); % the same as with reduction
end

