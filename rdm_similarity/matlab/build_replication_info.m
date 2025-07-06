function [replicate_ids, block_start_stop, block_labels] = build_replication_info(task_ids)
    uniq_tasks = unique(task_ids, 'stable');
    counts = histc(task_ids, uniq_tasks);
    lcm_count = lcm_many(counts);

    replicate_ids = [];
    block_start_stop = zeros(length(uniq_tasks), 2);
    block_labels = zeros(length(uniq_tasks), 1);
    offset = 0;

    for i = 1:length(uniq_tasks)
        task = uniq_tasks(i);
        count = counts(i);
        reps = lcm_count / count;
        indices = find(task_ids == task);

        replicated = reshape(repmat(indices(:)', reps, 1), [], 1);
        replicate_ids = [replicate_ids; replicated];

        block_start_stop(i, :) = [offset + 1, offset + length(replicated)];
        block_labels(i) = task;
        offset = offset + length(replicated);
    end
end

function out = lcm_many(v)
    out = v(1);
    for i = 2:length(v)
        out = lcm(out, v(i));
    end
end
