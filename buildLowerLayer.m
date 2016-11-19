function lower_layer = buildLowerLayer(gen_rules)
    num_rule = 2;
    num_category= size(gen_rules.W2,1);
    nlatent = num_rule*num_category;
    lower_layer = nnSetup([size(gen_rules.W1,3),nlatent]);
    lower_layer.W{1}(:) = 0;
    lower_layer.b{1}(:) = 0;

    %Set weights
    for i=1:num_category
        [~,idx] = sort(gen_rules.W2(i,:),'descend');
        idx = idx(1:num_rule);
        lower_layer.W{1}(((i-1)*num_rule+1):(i*num_rule),:) = squeeze(gen_rules.W1(i,idx,:));
        lower_layer.b{1}(((i-1)*num_rule+1):(i*num_rule)) = gen_rules.b1(i,idx);
    end
end
