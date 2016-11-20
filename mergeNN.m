function nn = mergeNN(lower_layer,upper_layer)
    nn = nnSetup([lower_layer.size(1), lower_layer.size(2), upper_layer.size(2)]);
    nn.W{1} = lower_layer.W{1};
    nn.W{2} = upper_layer.W{1};
    nn.b{1} = lower_layer.b{1};
    nn.b{2} = upper_layer.b{1};
end
