function [val grad ] = getFtypeCoefRegr(alpha, labels, gtLabels, regCoef) % note alpha is 0-based!!
  %alpha(alpha < 0) = 0;
  alpha = exp(alpha);
  % flatten all folds, remember reverse index
  allP = [];  %  p_ik (x_j)
  sumAlphaP = []; % sum_k [ alpha_k * p_ik (x_j) ]
  PforGtruth = []; % ln p_t_j (x_j)
  for fold = 1:length(labels)
    sumAlphaPFold = ones(length(labels{fold}), size(gtLabels{fold}, 2));
    allPFold = ones(length(alpha), size(gtLabels{fold}, 2), length(labels{fold}));
    PforGtruthFold = zeros(length(labels{fold}), length(alpha));
    for site = 1:length(labels{fold})
      %HACK
      if isempty(labels{fold}{site})
        labels{fold}{site} = zeros(0, size(gtLabels{fold}, 2)+1);
      end
      
      siteLabels = log(labels{fold}{site}(:,1:end-1));  
      siteFtypes = labels{fold}{site}(:,end)+1; % +1 because of 1-based indexing
      
      allPsite = zeros(length(alpha), size(siteLabels, 2));
      for ftypeIdx = 1:size(siteLabels, 1)
        allPsite(siteFtypes(ftypeIdx), :) = ...
          allPsite(siteFtypes(ftypeIdx), :) + siteLabels(ftypeIdx,:);
      end
      allPFold(:,:,site) = allPsite;
      
      PforGtruthFold(site, :) = allPsite(repmat(gtLabels{fold}(site,:) > 0.5, length(alpha), 1))';
      
      sumAlphaPFold(site,:) = sum(allPsite .* Embiggen(alpha), 1);
    end
      
    sumAlphaP = [sumAlphaP; sumAlphaPFold]; %#ok<AGROW>
    allP = cat(3, allP, allPFold); 
    PforGtruth = [PforGtruth; PforGtruthFold]; %#ok<AGROW>
  end  % TODO: move outside!
  
  expSumAlphaP = exp(sumAlphaP);
  
  regCoef = regCoef * size(sumAlphaP, 1);  % normalize to num factors
  
  val = -sum(sum(PforGtruth .* Embiggen(alpha'), 2) - log(sum(expSumAlphaP, 2))) ...
    + regCoef*sum(alpha);
  
  grad = -(sum(PforGtruth ...
              - sum(permute(allP, [3, 1, 2]) .* Embiggen(permute(expSumAlphaP, [1, 3, 2])), 3)...
                ./Embiggen(sum(expSumAlphaP, 2)))...
            - regCoef) ...
         .* alpha';
end

