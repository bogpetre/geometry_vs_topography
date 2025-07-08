% similar to the matlab built in squareform() method, except includes
% diagonal elements
function mat = squareform_with_diag(vec)
    n = floor(sqrt(2*length(vec)));
    mat = zeros(n);
    tril_idx = tril(true(n));
    mat(tril_idx) = vec;
    mat = mat + mat' - diag(diag(mat));
end