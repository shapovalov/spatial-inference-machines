classdef MSRCtester < handle
  %MSRCtester Runs tests on the MSRC (or other image segmentation) dataset
  
  properties
    infm = InferenceMachine(2)
    %infm = AutoContextMachine(2)
    samplers
    numRounds
  end
  
  methods
    % Constructs the class. Parameters:
    %  numRounds: number of training iterations
    %  reg_coef: (optional) Sets the regularization coefficient for the IM second stage.
    %    Set < 0 to skip tuning weights and set them to the uniform vector. 
    function self = MSRCtester(numRounds, reg_coef)
      self.numRounds = numRounds;
      if nargin > 1
        self.infm.setRegCoef(reg_coef);
      end
    end
    
    % Runs training and test on a given train/test split. Parameters:
    %   namesFile: the file that contains image names from new line each.
    %   nodeDir: directory where the .mat files with features are stored.
    %     The file name should be "<nodeDir>/features_<image_name>.mat"
    %     Each file contains the matrices: wordHist; colorHist; locHist.
    %   segDir: directory where the .mat files with segmentation maps are stored.
    %     The file name should be "<segDir>/struct_<image_name>.mat"
    %   splitFile: has the same length as namesFile, reads 1 for train, 2 for valid, 3 for test
    %   labFile: contains labels for all superpixels
    function [accuracy trainLabels testLabels] = traintest(self, namesFile, nodeDir, segDir, splitFile, labFile)
      LABELMAP = [0, 1:4, 0, 5:6, 0, 7:21];
        
      fid = fopen(namesFile);
      nm = textscan(fid, '%d %s');
      fclose(fid);
      fnames = nm{2};
      
      split = dlmread(splitFile); % 1 for train, 3 for test
      split = split(split ~= 2);  % ignore validation
      labels_blob = dlmread(labFile);
      
      unaries = cell(length(split), 1);
      gtlabels = cell(length(split), 1);
      segmaps = cell(length(split), 1);
      for i = 1:length(split)
        imlab = labels_blob(labels_blob(:,1) == i, :);
        assert(all(imlab(:,2) == (1:size(imlab,1))'));
        imlab = LABELMAP(imlab(:,3))';
        nonvoidMask = imlab > 0;
        gtlabels{i} = imlab(nonvoidMask);
        
        % load featyres
        imFtStruct = load([nodeDir, 'features_', fnames{i}(1:end-4), '.mat']);
        imfeatures = single([imFtStruct.wordHist; imFtStruct.colorHist; imFtStruct.locHist]);
        imfeatures = imfeatures(:, nonvoidMask);
        unaries{i} = imfeatures';
        
        % load segmentation map
        imSegStruct = load([segDir, 'struct_', fnames{i}(1:end-4), '.mat']);
        segmap = imSegStruct.baseRegions;
        newSegInd = cumsum(nonvoidMask);
        newSegInd(~nonvoidMask) = 0;
        segmap = newSegInd(segmap);
        segmaps{i} = segmap;
      end
      
      train.unaries = unaries(split == 1);
      train.gtlabels = gtlabels(split == 1);
      train.segmaps = segmaps(split == 1);
      test.unaries = unaries(split == 3);
      test.gtlabels = gtlabels(split == 3);
      test.segmaps = segmaps(split == 3);
      unaries = []; %#ok<NASGU>
      segmaps = []; %#ok<NASGU>
      
      % TEMP cut for testing
%       train.unaries = train.unaries(1:100);
%       train.gtlabels = train.gtlabels(1:100);
%       train.segmaps = train.segmaps(1:100);
%       test.unaries = test.unaries(21:28);
%       test.gtlabels = test.gtlabels(21:28);
%       test.segmaps = test.segmaps(21:28);
      
      ifs = ImageFactorSampler(train, test, 6);
      self.samplers = cellfun(@(x) ifs.copy(), cell(1,self.numRounds), 'UniformOutput', false);
      [trainLabels testLabels] = self.infm.train(self.samplers);
      save('MSRC/currLabels.mat', 'trainLabels', 'testLabels');
    end
  
  end
  
end

