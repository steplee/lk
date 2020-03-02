import torch; import clk
t = torch.randn(2,1,1024,1024)

clk.registerAffine(t)
clk.registerAffine(t)
clk.registerAffine(t)
clk.registerAffine(t)
clk.registerAffine(t)
