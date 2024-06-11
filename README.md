# Add MLP's learnable parameters to Kolmogorov-Arnold Network
Turn  $$\phi(x) = w_b b(x) + w_s \text{spline}(x).$$  into  $$\phi(x) = w_s \text{spline}(w_xx).$$
Based on efficient_KAN and KAN_Convolution, the learnable parameters of nodes in MLP are merged into KAN_linear, and the residual layer is removed. The accuracy on mnist is slightly improved.

---------------------------------------------------------------------------------------------------------------------------------------------
## Update
The weights $w_x$ can now scale the x dimension arbitrarily. This results in a large reduction in the number of parameters for KAN and a slight increase in accuracy.
