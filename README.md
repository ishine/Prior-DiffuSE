# Using Diffusion Models for Speech Enhancement

- [ ] 观察基于Complex Spectrum的范围是否在-1-1之间
- [ ] 实现Deblurring via Stochastic Refinement同样的方法





2.2 

## TODO:

1. 调整DDPM超参数（现在一个epoch就收敛了）
   1. lr: 1e-3 -> 2e-4
   2. lamdba: actually equals to learning rate at beginning
   3. x_init detached
   4. [ ] loss function
   5. fast sampling: inference_noise_schedule0.5 -> 0.35 noise_schedule  0.05 -> 0.035 
2. create function which return X_init after model_dic, X - X_init and X after model_ddpm.
   1. [ ] draw_spectrum added
3. create function which return X_t in model ddpm.

4. create function for drawing LS-MAE against Iteration.
5. added x_init metrics computing