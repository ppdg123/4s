function [nn_2stage] = trainCategoryNNTowStage(features,labels,param)
        
	%Step 1: Find generative rules for each category.
	gen_rules = findGenerativeRules(features,labels,param);
        
        %Step 2: Build the input->latent layer of stage 1 NN
        lower_layer = buildLowerLayer(gen_rules);
        
        %Step 3: Training the model weights of upper layer.
        upper_layer = trainUpperLayer(lower_layer,features,labels,param);
        
	%Step 4: Merging upper and lower layer.
        stage1 = mergeNN(lower_layer,upper_layer);
        nn_2stage.stage1 = stage1;
        
        %Step 5: Threshold & data filtering
        [ths,stds] = thresholdOfStage1(stage1,features,labels);
        nn_2stage.ths = ths;
        nn_2stage.stds = stds;
        [features_filtered,labels_filtered] = categoryFilterStage1(stage1,features,labels,ths,stds);
        
        %Step 6: Training Stage2 NN
        stage2 = trainCategoryNNStage2(features_filtered,labels_filtered,param);
        nn_2stage.stage2nn = stage2;
end

