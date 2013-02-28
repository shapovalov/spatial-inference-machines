classdef Classifier < handle
  %CLASSIFIER Interface for generic trainable classifier
  methods (Abstract)
    train(self, features, labels)
    labels = classify(self, features)
  end
end

