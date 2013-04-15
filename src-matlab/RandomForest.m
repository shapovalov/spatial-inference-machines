classdef RandomForest < Classifier 
  %LOGITREGRESSION the wrapper for logistic regression from Statistics toolbox
  
  properties (Constant)
    NUM_TREES = 100
    MIN_LEAF_SIZE = 1
  end
  
  properties (SetAccess = private, GetAccess = private)
    model
    numClasses  % useful when the latest class(es) is not represented in train set
  end
  
  properties (SetAccess = private)
    isTrained = false
    meanDepth
  end
  
  methods
    function self = RandomForest()
      addpath('randomforest-matlab/RF_Class_C');
    end
    
    function train(self, features, labels, tmpFold)
      assert(size(features, 1) == size(labels, 1));
      self.numClasses = size(labels, 2);
      
      %TEMP
      if ~all(any(labels))
        disp(any(labels));
      end
      
      % HACK to prevent hanging when all training instances are equal
      if all(all(bsxfun(@eq, features, features(1,:))))
        features(1,:) = features(1,:) - 1e-5;
      end
      
      % RF supports only homogenous instances
      assert(all(sum(labels, 2) == 1) && all(sum(labels == 1, 2) == 1));
      [rows cols] = find(labels);
      [rows idx] = sort(rows);   cols = cols(idx);
      assert(all(rows == (1:length(rows))'));
      
      % TEMP oversampling
      oldf = features; 
      oldc = cols;
      features = zeros(0, size(features, 2));
      cols = [];
      topclnum = max(histc(oldc, unique(oldc)));
      for cl = unique(oldc)'
        clIdx = (oldc == cl);
        repFactor = round(topclnum / sum(clIdx));
        features = [features; repmat(oldf(clIdx,:), repFactor, 1)];
        cols = [cols; repmat(oldc(clIdx), repFactor, 1)];
      end
      assert(length(cols) == size(features,1));
      
      % TEMP for failure debug
      %save(sprintf('tmpRFinput%d.mat', tmpFold), 'features', 'cols', 'oldf', 'oldc');
      
      %thispwd = pwd; cd('randomforest-matlab/RF_Class_C');  % may be slow?
      extra_options.nodesize = self.MIN_LEAF_SIZE;
      self.model = classRF_train(features, cols, self.NUM_TREES, ...
        floor(sqrt(size(features, 2))), extra_options);
        %size(features, 2), extra_options);
      %cd(thispwd);
      self.isTrained = true;
      
      [~, i] = max(double(self.model.treemap>0) .* ...
        repmat((1:size(self.model.treemap,1))', 1, self.NUM_TREES*2));
      self.meanDepth = mean(log2(i));
    end
    
    function labels = classify(self, features)
      assert(self.isTrained);
      
      %thispwd = pwd; cd('randomforest-matlab/RF_Class_C');
      [~, votes] = classRF_predict(features, self.model);
      %cd(thispwd);
      
      if size(votes, 2) ~= self.numClasses  % classes not represented in training set
        votes = [votes zeros(size(votes,1), self.numClasses - size(votes, 2))];
      end
      
      assert(size(votes, 2) == self.numClasses);
      assert(all(sum(votes,2) > 0));
      labels = bsxfun(@rdivide, votes, sum(votes,2));
    end
  end
  
end

