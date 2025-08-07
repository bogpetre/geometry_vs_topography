function [B, b] = mean_xsubject_B(x,y,varargin)
    b = zeros(size(x,1),2);
    for i = 1:size(x,1)
        this_x = x(i,:);
        this_y = y(i,:);

        good_ind = ~isnan(this_x) & ~isnan(this_y);
        this_x = this_x(good_ind);
        this_y = this_y(good_ind);

        if strcmp(varargin,'std')
            this_x = zscore(this_x);
            this_y = zscore(this_y);
        end
            
        X = [this_x(:), ones(length(this_x),1)];
        Y = this_y(:);

        b(i,:) = (X'*X)\X'*Y;
    end
    B = nanmean(b);
end