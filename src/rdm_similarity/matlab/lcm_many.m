function result = lcm_many(v)
    result = v(1);
    for i = 2:length(v)
        result = lcm(result, v(i));
    end
end
