function upper_layer = trainUpperLayer(lowernn,features,labels,param)
	num_category = size(labels,2);
	num_epochs = param.num_epochs;
	batch_size = param.batch_size;
	lambda = param.lambda;
	alpha = param.alpha;

	%Bottom-up forward
        forward_train_feature = forwardFeat(lowernn,features);
        
	%initialization
	upper_layer = nnSetup([lowernn.size(end),num_category]);
        upper_layer.lambda = lambda;
        upper_layer.alpha = alpha;
        opts.num_epochs = num_epochs;
        opts.batch_size = batch_size;
        
	%Training
	upper_layer = nnTrain(upper_layer,forward_train_feature,labels,opts);
end

function feat  = forwardFeat(net,x)
    net.evaling = 1;
    feat = [];
    for i = 1 : size(x, 1)
        %  feedforward
        f = nnForward(net, x(i, :));
        feat = [feat;f];
    end
end
