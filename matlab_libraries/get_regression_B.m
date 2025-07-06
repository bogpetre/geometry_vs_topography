% a simple functoin to get a univariate reg coef for use by bootci
% B = get_regression_B([x,y])
function B = get_regression_B(x)
    X = x(:,1)-mean(x(:,1));
    Y = x(:,2)-mean(x(:,2));

    B = (X'*X)\X'*Y;
end