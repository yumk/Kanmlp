#Add MLP's learnable parameters to Kolmogorov-Arnold Network
Turn $\phi(x) = w_b b(x) + w_s \text{spline}(x).$ into  $\phi(x) = w_s \text{spline}(w_xx).$ 
Based on efficient_KAN and KAN_Convolution, the learnable parameters of nodes in MLP are merged into KANlinear, and the residual layer is removed, and the accuracy on mnist is slightly improved.
