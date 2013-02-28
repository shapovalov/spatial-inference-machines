function [val grad ] = getFtypeCoefLkhood(alpha, labels, gtLabels, regCoef) % note alpha is 0-based!!
  %alpha(alpha < 0) = 0;
  alpha = exp(alpha);
  % flatten all folds, remember reverse index
  objLabelsAlpha = [];  % sum_i prod_k ( p_ik ^alpha_k (x_j) )
  objLabelsAlphaLnP = []; % sum_i [ prod_k ( p_ik ^alpha_k (x_j) ) * ln p_ik (x_j)]
  objLabelsAlphaLnPLnP = []; % sum_i [ prod_k ( p_ik ^alpha_k (x_j) ) * ln p_ik (x_j) * ln p_il (x_j)]
  lnPforGtruth = []; % ln p_t_j (x_j)
  for fold = 1:length(labels)
    objLabelsAlphaFold = ones(length(labels{fold}), size(gtLabels{fold}, 2));
    objLabelsAlphaLnPFold = ones(length(labels{fold}), length(alpha));
    objLabelsAlphaLnPLnPFold = ones(length(alpha), length(alpha), length(labels{fold}));
    lnPforGtruthFold = zeros(length(labels{fold}), length(alpha));
    for site = 1:length(labels{fold})
      %HACK
      if isempty(labels{fold}{site})
        labels{fold}{site} = zeros(0, size(gtLabels{fold}, 2)+1);
      end
      
      siteLabels = labels{fold}{site}(:,1:end-1);
      siteFtypes = labels{fold}{site}(:,end)+1; % +1 because of 1-based indexing
      
      objLabelsAlphaFold(site,:) = ... 
        prod(bsxfun(@power, siteLabels, alpha(siteFtypes)));
      
      objLabelsAlphaLnPSite = zeros(length(alpha), size(siteLabels, 2));
      for ftypeIdx = 1:size(siteLabels, 1)
        objLabelsAlphaLnPSite(siteFtypes(ftypeIdx), :) = ...
          objLabelsAlphaLnPSite(siteFtypes(ftypeIdx), :) + log(siteLabels(ftypeIdx,:));
      end
      objLabelsAlphaLnPFold(site, :) = ...
        sum(bsxfun(@times, objLabelsAlphaFold(site,:), objLabelsAlphaLnPSite), 2);
      
      objLabelsAlphaLnPLnPSite = repmat(permute(objLabelsAlphaLnPSite, [1, 3, 2]), [1,length(alpha),1]) .* ...
        Embiggen(permute(objLabelsAlphaLnPSite, [3, 1, 2]));
      objLabelsAlphaLnPLnPFold(:, :, site) = ...
        sum(objLabelsAlphaLnPLnPSite .* Embiggen(permute(objLabelsAlphaFold(site,:), [1 3 2])), 3);
                 
      gtlab = log(siteLabels(:,gtLabels{fold}(site,:) > 0.5));
      assert(length(gtlab) == size(siteLabels, 1));
      for ftypeIdx = 1:size(siteLabels, 1)
        lnPforGtruthFold(site, siteFtypes(ftypeIdx)) = ...
          lnPforGtruthFold(site, siteFtypes(ftypeIdx)) + gtlab(ftypeIdx);
      end
    end
      
    objLabelsAlpha = [objLabelsAlpha; objLabelsAlphaFold]; %#ok<AGROW>
    objLabelsAlphaLnP = [objLabelsAlphaLnP; objLabelsAlphaLnPFold]; %#ok<AGROW>
    objLabelsAlphaLnPLnP = cat(3, objLabelsAlphaLnPLnP, objLabelsAlphaLnPLnPFold); 
    lnPforGtruth = [lnPforGtruth; lnPforGtruthFold]; %#ok<AGROW>
  end  % TODO: move outside!
  
  gtLabelsFlat = cell2mat(gtLabels'); 
  assert(all(size(gtLabelsFlat) == size(objLabelsAlpha)));
  assert(all(sum(gtLabelsFlat > 0.5, 2) == 1));
  
  partition = sum(objLabelsAlpha, 2);
  objLabelsAlpha = objLabelsAlpha'; % for correcr 1D indexing
  
  regCoef = regCoef * size(partition, 1);  % normalize to num factors
  
   %val = -sum(log(objLabelsAlpha(gtLabelsFlat' > 0.5)) - log(partition+ 1e-300));
%   disp(mean(objLabelsAlpha(gtLabelsFlat' > 0.5) ./ partition));
  val = -sum(objLabelsAlpha(gtLabelsFlat' > 0.5) ./ (partition+ 1e-300)) + regCoef*sum(alpha);
  
  %grad = -(sum(lnPforGtruth) - sum(objLabelsAlphaLnP ./ Embiggen(partition+ 1e-300))).*alpha';
   grad = -(sum((lnPforGtruth .* Embiggen(partition) - objLabelsAlphaLnP) ...
       .* Embiggen(objLabelsAlpha(gtLabelsFlat' > 0.5) ./(partition.^2 + 1e-300))) - regCoef)...
     .*alpha'; %because of exp
  assert(all(~isnan(grad)));
  
%   hess = zeros(length(alpha), length(alpha));
%   for j = 1:size(gtLabelsFlat, 1)
%     hess = hess - objLabelsAlphaLnPLnP(:,:,j) ./ partition(j) + ...
%       objLabelsAlphaLnP(j,:)' * objLabelsAlphaLnP(j,:) ./ partition(j).^2;
%   end
%   hess = -hess; % maximize 
end

