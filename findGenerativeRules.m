function [gen_rules] = findGenerativeRules(features,labels,param)
	num_latent = param.num_latent;
	num_epochs = param.num_epochs;
        num_train = param.num_train;
	num_subnet = param.num_subnet;
        opts = param.opts;
	lambda = param.lambda;
	alpha = param.alpha;
	batch_size = param.batch_size;

        for i=1:num_subnet
	    %Random selecting
            idx = randperm(size(features,1));
            train_idx = idx(1:num_train);
            train_features = features(train_idx,:);
            train_labels = labels(train_idx,:);

	    %Initializing subnets
            subnet = nnSetup([size(features,2),num_latent,size(labels,2)]);
            subnet.lambda = lambda;
            subnet.alpha = alpha;
            opts.num_epochs = num_epochs;
            opts.batch_size = batch_size;

	    %Training subnets
            subnet = nnTrain(subnet,train_features,train_labels,opts);

	    %Selecting generative rules
            for j=1:size(labels,2)
                [val,idx] = max(subnet.W{2}(j,:));
                gen_rules.contro(j,i) = val;
                gen_rules.W2(j,i) = subnet.W{2}(j,idx);
                gen_rules.W1(j,i,:) = subnet.W{1}(idx,:);
                gen_rules.b1(j,i) = subnet.b{1}(idx);
                gen_rules.b2(j,i) = subnet.b{2}(j);
            end
        end
end
