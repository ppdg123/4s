function nn = mergeNN(lowernn,uppernn)
    nn = nnsetup([lowernn.size(1), lowernn.size(2), uppernn.size(2)]);
    nn.W{1} = lowernn.W{1};
    nn.W{2} = uppernn.W{1};
    nn.b{1} = lowernn.b{1};
    nn.b{2} = uppernn.b{1};
end