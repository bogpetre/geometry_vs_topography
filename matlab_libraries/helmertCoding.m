function helmertMatrix = helmertCoding(categories)
    % Convert to unique levels
    uniqueCategories = unique(categories);
    k = length(uniqueCategories);
    
    % Initialize Helmert matrix
    helmertMatrix = zeros(length(categories), k - 1);
    
    % Generate Helmert contrasts
    for i = 1:k-1
        scale = sum(uniqueCategories >= uniqueCategories(i));
        contrast = (categories == uniqueCategories(i)) - ...
                   (categories > uniqueCategories(i));
        contrast(contrast > 0) = contrast(contrast > 0) * (scale - 1);
        helmertMatrix(:, i) = contrast/scale;
    end
end