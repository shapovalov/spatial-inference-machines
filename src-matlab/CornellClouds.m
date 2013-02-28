classdef CornellClouds < handle
  %CORNELLCLOUDS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties (Constant)
    NUM_FOLDS = 4;
  end
  
  properties
    infm = InferenceMachine()
    samplers
    numRounds
  end
  
  methods
    function self = CornellClouds(numRounds, reg_coef)
      self.numRounds = numRounds;
      if nargin > 1
        self.infm.setRegCoef(reg_coef);
      end
    end
    
    function [accuracy labels] = xval(self, nodesFile, edgesFile, labelmapFile, pcdPathMask)
      nodesRaw = dlmread(nodesFile, '', 55, 0);
      edgesRaw = dlmread(edgesFile, '', 49, 0);
      % map labels to 1..m
      labelmap = dlmread(labelmapFile);
      [row col] = find(bsxfun(@eq, nodesRaw(:,3)', labelmap(:,1)));
      %assert(all(col == 1:size(nodesRaw, 1)));
      newlabs = zeros(1, size(nodesRaw, 1));
      newlabs(col) = labelmap(row,2);
      nodesRaw(:,3) = newlabs;
      nodesRaw = nodesRaw(newlabs ~= 0,:);
      
      edgCol = any(bsxfun(@eq, edgesRaw(:,4)', labelmap(:,1)), 1) & ...
        any(bsxfun(@eq, edgesRaw(:,5)', labelmap(:,1)), 1); % both labels ok
      edgesRaw = edgesRaw(edgCol,:);
      
      sceneIdx = unique(nodesRaw(:,1));
      assert(all(sort(unique(edgesRaw(:,1))) == sort(sceneIdx)));
      assert(mod(length(sceneIdx), self.NUM_FOLDS) == 0); % might be relaxed
      
      rightAnsw = [];
      labels = cell(1, self.numRounds);
      for fold = 1:self.NUM_FOLDS
        testScIdx = any(bsxfun(@eq, nodesRaw(:,1), sceneIdx(fold:self.NUM_FOLDS:end)'), 2);
        edgeTestScIdx = any(bsxfun(@eq, edgesRaw(:,1), sceneIdx(fold:self.NUM_FOLDS:end)'), 2);
        
        train.nodesFtLb = nodesRaw(~testScIdx, :);
        test.nodesFtLb = nodesRaw(testScIdx, :);
        train.edgesFtLb = edgesRaw(~edgeTestScIdx, :);
        test.edgesFtLb = edgesRaw(edgeTestScIdx, :);
        
        % TEMP for fast test
%         train.nodesFtLb = train.nodesFtLb(train.nodesFtLb(:,1)<=3, :);
%         test.nodesFtLb = test.nodesFtLb(train.nodesFtLb(:,1)<=3, :);
%         train.edgesFtLb = train.edgesFtLb(train.nodesFtLb(:,1)<=3, :);
%         test.edgesFtLb = test.edgesFtLb(train.nodesFtLb(:,1)<=3, :);
        
        ccfs = CloudFactorSampler(train, test, pcdPathMask, 5); % test lb are not used
        % TEMP test on train set
        %ccfs = CloudFactorSampler(train, train, pcdPathMask);
        self.samplers = cellfun(@(x) ccfs.copy(), cell(1,self.numRounds), 'UniformOutput', false);
        testLabels = self.infm.train(self.samplers);
        rightAnsw = [rightAnsw; test.nodesFtLb(:,3)]; %#ok<AGROW>
        % TEMP test on train set
        %rightAnsw = [rightAnsw; train.nodesFtLb(:,3)]; %#ok<AGROW>
        
        for i = 1:self.numRounds
          labels{i} = [labels{i}; cell2mat(testLabels{i}(:))];  % % TODO: multiple rounds?
        end
      end
     
      accuracy = zeros(1, self.numRounds);
      for i = 1:self.numRounds
        answ_i = labels{i};
        accuracy(i) = sum(answ_i((1:size(answ_i, 1))' + size(answ_i, 1)*(rightAnsw-1))) / size(answ_i, 1);
      end
      
      disp(accuracy);
    end
    
    
    function labels = infer(self, nodesFile, edgesFile, labelmapFile, pcdPathMask)
      nodesRaw = dlmread(nodesFile, '', 55, 0);
      edgesRaw = dlmread(edgesFile, '', 49, 0);
      %TEMP
      %nodesRaw = nodesRaw(nodesRaw(:,1) == 4,:);
      %edgesRaw = edgesRaw(edgesRaw(:,1) == 4,:);
      % map labels to 1..m
      labelmap = dlmread(labelmapFile);
      [row col] = find(bsxfun(@eq, nodesRaw(:,3)', labelmap(:,1)));
      %assert(all(col == 1:size(nodesRaw, 1)));
      newlabs = zeros(1, size(nodesRaw, 1));
      newlabs(col) = labelmap(row,2);
      nodesRaw(:,3) = newlabs;
      
      test.nodesFtLb = nodesRaw;
      test.edgesFtLb = edgesRaw;
      
      for i = 1:self.numRounds, self.samplers{i}.setTestData(test, pcdPathMask); end
      tic
      labels = self.infm.infer(self.samplers);
      toc
      
      labels = cell2mat(labels');
    end
  end
  
end

