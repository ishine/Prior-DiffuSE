# Using Diffusion Models for Speech Enhancement

- [ ] 观察基于Complex Spectrum的范围是否在-1-1之间
- [ ] 实现Deblurring via Stochastic Refinement同样的方法





2.2 

## TODO list:

1. 调整DDPM超参数（现在一个epoch就收敛了）
   1. lr: 1e-3 -> 2e-4
   2. lamdba: actually equals to learning rate at beginning
   3. x_init detached
   4. loss function: l1 loss -> com_mse_loss
   5. fast sampling: inference_noise_schedule0.5 -> 0.35 noise_schedule  0.05 -> 0.035 
2. create function which return X_init after model_dic, X - X_init and X after model_ddpm.
   1. draw_spectrum added by arg --draw
3. create function for drawing LS-MAE against Iteration.
4. added x_init metrics computing

## get started

1. modifie parameter in param.py(config in learner code) and diff.yml(args in learner code)
2. train
   1. train dis model only
       modified code
   2. train ddpm model only 
      1. move the trained dis model's ckp into the new asset folder
      2. train
         ```python
         python main.py --asset <asset_name> --retrain
         ```
   3. joint train
      ```python
      python main.py --asset <asset_name> --joint
      ```
3. draw evaluated data include noisy audio, init_audio, predicted_audio, true_delta, predicted_delta
   ```python
      python main.py --asset <asset_name> -- retrain --draw
   ```
   
