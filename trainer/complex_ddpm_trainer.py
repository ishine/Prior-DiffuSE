import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import utils.params
from utils import *
from model import *
# from model.diff2 import DiffWave
from model.diff3 import DiffUNet1
import logging
import wandb
from tqdm import tqdm
from scripts.draw_spectrum import plot_stft
import matplotlib.pyplot as plt
from plotnine import *
import pandas as pd

wandb.init(project="ddpm")


class ComplexDDPMTrainer(object):
    def __init__(self, args, config):
        # config
        self.args = deepcopy(args)
        self.config = deepcopy(config)
        self.params = utils.params.params
        self.step = 0
        self.loss_fn_eva = nn.L1Loss()      # use for evaluation of x0


        beta = np.array(self.params.noise_schedule)     # noise_schedule --> beta       noise_level --> alpha^bar
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))



        '''dataset & dataloader'''
        collate = Collate(self.config)
        self.tr_dataset = VBTrDataset('data/noisy_trainset_wav', 'data/clean_trainset_wav', config)
        cv_dataset = VBCvDataset('data/noisy_testset_wav', 'data/clean_testset_wav', config)
        logging.info(f'Total {self.tr_dataset.__len__()} train data.')  # 11572
        logging.info(f'Total {cv_dataset.__len__()} eval data.')  # 824
        self.tr_dataloader = DataLoader(self.tr_dataset,
                                        batch_size=self.config.train.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=os.cpu_count(),
                                        collate_fn=collate.collate_fn)
        self.cv_dataloader = DataLoader(cv_dataset,
                                        batch_size=self.config.train.batch_size,
                                        # batch_size=1,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=os.cpu_count(),
                                        collate_fn=collate.collate_fn)

        '''model'''
        self.model = eval(self.config.model.name)().cuda()
        self.model_ddpm = DiffUNet1(self.params).cuda()

        '''optimizer'''
        if self.config.optim.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                self.config.optim.lr,
                weight_decay=self.config.optim.l2
            )
            self.optimizer_ddpm = torch.optim.Adam(
                self.model_ddpm.parameters(),
                self.config.optim_ddpm.lr,
                weight_decay=self.config.optim_ddpm.l2
            )

        '''retrain'''
        if self.args.retrain:
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))
            self.model.load_state_dict(pretrained_data[0])
            self.optimizer.load_state_dict(pretrained_data[1])
            # self.model_ddpm.load_state_dict(pretrained_data[2])
            # self.optimizer_ddpm.load_state_dict(pretrained_data[3])

        '''logger'''
        wandb.watch(self.model, log="all")

    def inference_schedule(self, fast_sampling=False):
        """
        Compute fixed parameters in ddpm

        :return:
            alpha:          alpha for training,         sizelike noise_schedule
            beta:           beta for inference,         sizelike inference_noise_schedule or noise_schedule
            alpha_cum:      alpha_cum for inference
            sigmas:         sqrt(beta_t^tilde)
            T:              Timesteps
        """
        training_noise_schedule = np.array(self.params.noise_schedule)
        inference_noise_schedule = np.array(
            self.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule    # alpha_t for train
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        sigmas = [0 for i in alpha]
        for n in range(len(alpha) - 1, -1, -1):
            sigmas[n] = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5      # sqrt(beta_t^tilde)
        # print("sigmas", sigmas)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                                talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                    T.append(t + twiddle)   # from schedule to T which as the input of model
                    break
        T = np.array(T, dtype=np.float32)
        idx = [i for i in range(len(alpha))]
        v_dict = {
            "idx": idx,
            "alpha": alpha,
            "beta": beta,
            "alpha_cum": alpha_cum,
            "sigmas": sigmas,
            "T": T
        }
        data_df = pd.DataFrame(v_dict)
        data_plot = pd.melt(data_df[["idx", "alpha","beta", "alpha_cum", "sigmas"]], id_vars=["idx"])   # melt turn col_name into feature
        print(data_plot)
        plot =  ggplot(data_plot) + geom_line(aes(x="idx", y="value", color='variable'))+ xlab("steps")
        plot.save("full_sample.pdf")

        return alpha, beta, alpha_cum, sigmas, T

    def train_ddpm(self, max_steps=None):
        torch.backends.cudnn.enabled = True

        '''variants initialize'''
        prev_cv_loss = float("inf")
        best_cv_loss = float("inf")
        cv_no_impv = 0
        harving = False

        '''training'''
        for epoch in range(self.config.train.n_epochs):
            alpha, beta, alpha_cum, sigmas, T = self.inference_schedule(fast_sampling=self.params.fast_sampling)
            exit()
            logging.info(f'Epoch {epoch}')
            self.model_ddpm.train()
            # self.model.train()
            for features in tqdm(self.tr_dataloader):
                if max_steps is not None and self.step >= max_steps:
                    return
                loss = self.train_step(features)
                self.step += 1
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')

            '''evaluation after an epoch'''
            self.model.eval()
            self.model_ddpm.eval()
            all_loss_list = []
            all_csig_list, all_cbak_list, all_covl_list, all_pesq_list, all_ssnr_list, all_stoi_list = [], [], [], [], [], []
            # all_loss_list_init = []
            # all_csig_list_init, all_cbak_list_init, all_covl_list_init, all_pesq_list_init, all_ssnr_list_init, all_stoi_list_init = [], [], [], [], [], []
            alpha, beta, alpha_cum, sigmas, T = self.inference_schedule(fast_sampling=self.params.fast_sampling)

            with torch.no_grad():
                for batch in tqdm(self.cv_dataloader):
                    batch_feat = batch.feats.cuda()
                    batch_label = batch.labels.cuda()
                    # print(batch_label)
                    '''four approaches for feature compression'''
                    noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])  # [B, 1, T, F]
                    clean_phase = torch.atan2(batch_label[:, -1, :, :], batch_label[:, 0, :, :])

                    if self.config.train.feat_type == 'normal':
                        batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
                    elif self.config.train.feat_type == 'sqrt':
                        batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                            torch.norm(batch_label, dim=1)) ** 0.5
                    elif self.config.train.feat_type == 'cubic':
                        batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                            torch.norm(batch_label, dim=1)) ** 0.3
                    elif self.config.train.feat_type == 'log_1x':
                        batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                                  torch.log(torch.norm(batch_label, dim=1) + 1)
                    if self.config.train.feat_type in ['normal', 'sqrt', 'cubic', 'log_1x']:
                        batch_feat = torch.stack(
                            (batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                            # [B, 2, T, F]
                            dim=1)
                        batch_label = torch.stack(
                            (batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                            dim=1)

                    '''evaluation'''
                    init_audio = self.model(batch_feat)  # [B, 2, T, F]
                    audio = torch.randn_like(init_audio)                                                        # XT = N(0, I),  [N, 2, T, F]
                    # print(audio)
                    # print(audio.shape)
                    # print("________________________________________________________________")
                    # print(audio)
                    # exit()
                    N = audio.shape[0]
                    gamma = [0 for i in alpha]                                                                  # the first 2 num didn't use
                    for n in range(len(alpha)):
                        gamma[n] = sigmas[n]                                                                    # beta^tilde
                    # print(gamma)    #[0.715, 0.0095, 0.031, 0.095, 0.220, 0.412]
                    gamma[0] = 0.2                                                                              # beta^tilde_0 = 0.2
                    # print("gamma",gamma)
                    for n in range(len(alpha) - 1, -1, -1):
                        c1 = 1 / alpha[n] ** 0.5                                                                # c1 in mu equation
                        c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5                                                # c2 in mu equation
                        # print("T[n]", torch.tensor([T[n]], device=audio.device).repeat(N).shape)
                        # print("T", torch.randint(0, len(self.params.noise_schedule), [N], device=device).shape)
                        # print("audio", audio.shape)
                        predicted_noise = self.model_ddpm(audio, init_audio,                                    # z_theta(x_t, condition, t)
                                                          torch.tensor([T[n]], device=audio.device).repeat(N))
                        # audio = c1 * ((1-gamma[n])*mu+gamma[n]* (noisy_audio-init_audio))  # 插值
                        audio = c1 * (audio - c2 * predicted_noise)                                             # mu_theta(x_t, z_theta)
                        # print("________________________________________________________________")
                        # print(predicted_noise)
                        # print("________________________________________________________________")
                        # print("predicted_noise", predicted_noise.shape)
                        # print("c1", c1)
                        # print("c2", c2)
                        # print("predicted_noise", predicted_noise)
                        # print(audio)
                        if n > 0:
                            noise = torch.randn_like(audio)
                            sigma = gamma[n]                                                                    # sigma = gamma[n]
                            # newsigma = max(0, sigma - c1 * gamma[n])                                            # ???
                            audio += sigma * noise                                                           # x_t-1 = mu_theta + beta^tilde * epsilon

                        # audio = torch.clamp(audio, -1.0, 1.0) # Diffuse used after preprocess
                    # audio += init_audio

                    '''plot x0'''
                    # for i in range(N):
                    #     # print("t", t[i])
                    #     print("________________________________________________________________")
                    #     print("batch_label", torch.max(batch_label), torch.min(batch_label))
                    #     plot_stft(batch_label[i].cpu(), features.wav_len_list[i], title="batch_label")
                    #     print("________________________________________________________________")
                    #     # print("noisy_audio", torch.max(noisy_audio), torch.min(noisy_audio))
                    #     # plot_stft(noisy_audio[i].cpu(), features.wav_len_list[i], title="noisy_audio")
                    #     print("________________________________________________________________")
                    #     print("init_audio", torch.max(init_audio), torch.min(init_audio))
                    #     plot_stft(init_audio[i].cpu(), features.wav_len_list[i], title="init_audio")
                    #     print("________________________________________________________________")
                    #     print("predicted", torch.max(audio), torch.min(audio))
                    #     plot_stft(audio[i].detach().cpu(), features.wav_len_list[i], title="predicted")
                    #     print("________________________________________________________________")
                    #     # print("loss_ddpm", loss_ddpm)
                    #     print("________________________________________________________________")

                    '''metrics compute'''
                    # x
                    batch_loss = com_mse_loss(audio, batch_label ,batch.frame_num_list)
                    # print(audio)
                    # print(batch_label)
                    # print(batch.frame_num_list)
                    batch_result = compare_complex(audio, batch_label, batch.frame_num_list,
                                                   feat_type=self.config.train.feat_type)  # compute evaluate metrics
                    all_loss_list.append(batch_loss.item())
                    all_csig_list.append(batch_result[0])
                    all_cbak_list.append(batch_result[1])
                    all_covl_list.append(batch_result[2])
                    all_pesq_list.append(batch_result[3])
                    all_ssnr_list.append(batch_result[4])
                    all_stoi_list.append(batch_result[5])

                    # x_init
                    # batch_loss = self.loss_fn(batch_label, init_audio)
                    # batch_result = compare_complex(init_audio, batch_label, batch.frame_num_list,
                    #                                feat_type=self.config.train.feat_type)  # compute evaluate metrics
                    # all_loss_list_init.append(batch_loss.item())
                    # all_csig_list_init.append(batch_result[0])
                    # all_cbak_list_init.append(batch_result[1])
                    # all_covl_list_init.append(batch_result[2])
                    # all_pesq_list_init.append(batch_result[3])
                    # all_ssnr_list_init.append(batch_result[4])
                    # all_stoi_list_init.append(batch_result[5])

                wandb.log(
                    {
                        'test_com_mse_loss': np.mean(all_loss_list),  # mean loss in val dataset
                        'test_mean_csig': np.mean(all_csig_list),
                        'test_mean_cbak': np.mean(all_cbak_list),
                        'test_mean_covl': np.mean(all_covl_list),
                        'test_mean_pesq': np.mean(all_pesq_list),
                        'test_mean_ssnr': np.mean(all_ssnr_list),
                        'test_mean_stoi': np.mean(all_stoi_list)
                        # 'test_mean_mse_loss_init': np.mean(all_loss_list_init),
                        # 'test_mean_csig_init': np.mean(all_csig_list_init),
                        # 'test_mean_cbak_init': np.mean(all_cbak_list_init),
                        # 'test_mean_covl_init': np.mean(all_covl_list_init),
                        # 'test_mean_pesq_init': np.mean(all_pesq_list_init),
                        # 'test_mean_ssnr_init': np.mean(all_ssnr_list_init),
                        # 'test_mean_stoi_init': np.mean(all_stoi_list_init)
                    }
                )

            cv_loss = np.mean(all_loss_list)
            '''Adjust the learning rate and early stop'''
            if self.config.optim.half_lr > 1:
                if cv_loss >= prev_cv_loss:
                    cv_no_impv += 1
                    if cv_no_impv == self.config.optim.half_lr:  # adjust lr depend on cv_no_impv
                        harving = True
                    if cv_no_impv >= self.config.optim.early_stop > 0:  # early stop
                        logging.info("No improvement and apply early stop")
                        break
                else:
                    cv_no_impv = 0

            if harving == True:
                optim_state = self.optimizer_ddpm.state_dict()
                for i in range(len(optim_state['param_groups'])):
                    optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                logging.info('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                harving = False
            prev_cv_loss = cv_loss

            if cv_loss < best_cv_loss:
                logging.info(
                    f"last best loss is: {best_cv_loss}, current loss is: {cv_loss}, save best_checkpoint.pth")
                best_cv_loss = cv_loss
                states = [
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    self.model_ddpm.state_dict(),
                    self.optimizer_ddpm.state_dict()
                ]
                torch.save(states, os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))

            '''save latest checkpoint'''
            states = [
                self.model.state_dict(),
                self.optimizer.state_dict(),
                self.model_ddpm.state_dict(),
                self.optimizer_ddpm.state_dict()
            ]
            torch.save(states, os.path.join(self.args.checkpoint, f'checkpoint_{epoch}.pth'))


    def train_step(self, features):
        self.optimizer_ddpm.zero_grad()
        # self.optimizer.zero_grad()
        batch_feat = features.feats.cuda()
        batch_label = features.labels.cuda()
        batch_frame_num_list = features.frame_num_list

        '''four approaches for feature compression'''
        noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :,
                                                           :])  # [B, 1, T, F] noisy_phase means <相成分>, batch_feat means <batch feature> ?
        clean_phase = torch.atan2(batch_label[:, -1, :, :],
                                  batch_label[:, 0, :, :])  # torch.atan2 means 双变量反正切函数,值域为（-pi, pi）
        if self.config.train.feat_type == 'normal':
            batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label,
                                                                                dim=1)  # [B, 1, T, F] <相应频率下的分量幅度>
        elif self.config.train.feat_type == 'sqrt':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                torch.norm(batch_label, dim=1)) ** 0.5
        elif self.config.train.feat_type == 'cubic':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                torch.norm(batch_label, dim=1)) ** 0.3
        elif self.config.train.feat_type == 'log_1x':
            batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                      torch.log(torch.norm(batch_label, dim=1) + 1)
        if self.config.train.feat_type in ['normal', 'sqrt', 'cubic', 'log_1x']:
            batch_feat = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                                     # [B, 2, T, F] <相应频率下的分量幅度在 实轴和虚轴的投影>
                                     dim=1)
            batch_label = torch.stack(
                (batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                dim=1)

        # print("batch_feat: ", batch_feat.shape)
        '''discriminative model'''
        init_audio = self.model(batch_feat).detach() # [8, 2, 301, 161]
        # 计算 model 参数量 model 总参数数量和：1662565 model_ddpm 总参数数量和：1258371
        # params = list(self.model.parameters())
        # k = 0
        # for i in params:
        #     l = 1
        #     print("该层的结构：" + str(list(i.size())))
        #     for j in i.size():
        #         l *= j
        #     print("该层参数和：" + str(l))
        #     k = k + l
        # print("model 总参数数量和：" + str(k))
        # params = list(self.model_ddpm.parameters())
        # k = 0
        # for i in params:
        #     l = 1
        #     print("该层的结构：" + str(list(i.size())))
        #     for j in i.size():
        #         l *= j
        #     print("该层参数和：" + str(l))
        #     k = k + l
        # print("model_ddpm 总参数数量和：" + str(k))
        # exit(0)

        # loss_dis = eval(self.config.train.loss)(init_audio, batch_label, batch_frame_num_list)
        loss_dis = 0

        '''ddpm'''

        N  = batch_feat.shape[0]  # Batch size
        device = batch_feat.device
        self.noise_level = self.noise_level.to(device)                                  # alpha_bar     [1000]

        t = torch.randint(0, len(self.params.noise_schedule), [N], device=device)
        # print("t:", t)
        noise_scale = self.noise_level[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)        # alpha_bar_t       [N, 1, 1, 1]
        noise_scale_sqrt = noise_scale ** 0.5                                           # sqrt(alpha_bar_t) [N, 1, 1, 1]
        noise = torch.randn_like(batch_feat)                                            # epsilon           [N, 2, T, F]
        # print("batch_label: ", batch_label.shape)
        # print("noise: ", noise.shape)
        # print("noise_scale_sqrt: ", noise_scale_sqrt.shape)
        # _ = noise_scale_sqrt * batch_label
        # exit()
        # print("batch_label-init_audio:", batch_label-init_audio)
        # noisy_audio = noise_scale_sqrt * (batch_label-init_audio) + (1.0 - noise_scale) ** 0.5 * noise  # pirorgrad
        noisy_audio = noise_scale_sqrt * (batch_label) + (1.0 - noise_scale) ** 0.5 * noise
        predicted = self.model_ddpm(noisy_audio, init_audio, t)  # epsilon^hat
        # for i in range(N):
        #     print("t", t[i])
        #     print("________________________________________________________________")
        #     print("batch_label", torch.max(batch_label), torch.min(batch_label))
        #     # plot_stft(batch_label[i].cpu(), features.wav_len_list[i], title="batch_label")
        #     print("________________________________________________________________")
        #     print("noise_scale_sqrt", noise_scale_sqrt)
        #     print("noisy_audio", torch.max(noisy_audio), torch.min(noisy_audio))
        #
        #     plt.matshow(noisy_audio[i][0].cpu().numpy())
        #     plt.colorbar()
        #     plt.show()
        #     # plot_stft(noisy_audio[i].cpu(), features.wav_len_list[i], title="noisy_audio")
        #     print("________________________________________________________________")
        #     print("init_audio", torch.max(init_audio), torch.min(init_audio))
        #     # plot_stft(init_audio[i].cpu(), features.wav_len_list[i], title="init_audio")
        #     print("________________________________________________________________")
        #     print("predicted", torch.max(predicted), torch.min(predicted))
        #     # plot_stft(predicted[i].detach().cpu(), features.wav_len_list[i], title="predicted")
        #     print("________________________________________________________________")
        #     # print("loss_ddpm", loss_ddpm)
        #     print("________________________________________________________________")

        # loss_ddpm = self.loss_fn_eva(noise, predicted)
        loss_ddpm = eval(self.config.train.loss)(predicted, noise , batch_frame_num_list)

        # loss = self.config.train.lam * loss_ddpm + loss_dis
        loss = loss_ddpm

        wandb.log(
            {
             # 'dis_loss': loss1.item(),
             'ddpm_loss': loss_ddpm.item(),
             # 'loss_sum': loss.item()
            }
        )
        # loss = loss1
        # wandb.log(
        #     {'loss_sum': loss.item()}
        # )
        loss.backward()
        # self.optimizer.step()
        self.optimizer_ddpm.step()


        return loss


    def train(self):
        prev_cv_loss = float("inf")
        best_cv_loss = float("inf")
        cv_no_impv = 0
        harving = False
        for epoch in range(self.config.train.n_epochs):
            logging.info(f'Epoch {epoch}')
            self.model.train()
            '''train'''
            for batch in tqdm(self.tr_dataloader):
                self.optimizer.zero_grad()
                batch_feat = batch.feats.cuda()
                batch_label = batch.labels.cuda()
                noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])  # [B, 1, T, F] noisy_phase means <相成分>, batch_feat means <batch feature> ?
                clean_phase = torch.atan2(batch_label[:, -1, :, :], batch_label[:, 0, :, :])    # torch.atan2 means 双变量反正切函数,值域为（-pi, pi）

                '''four approaches for feature compression'''
                if self.config.train.feat_type == 'normal':
                    batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1) # [B, 1, T, F] <相应频率下的分量幅度>
                elif self.config.train.feat_type == 'sqrt':
                    batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, ( # 范数的平方根？
                        torch.norm(batch_label, dim=1)) ** 0.5
                elif self.config.train.feat_type == 'cubic':
                    batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                        torch.norm(batch_label, dim=1)) ** 0.3
                elif self.config.train.feat_type == 'log_1x':
                    batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                              torch.log(torch.norm(batch_label, dim=1) + 1)
                if self.config.train.feat_type in ['normal', 'sqrt', 'cubic', 'log_1x']:
                    batch_feat = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),    # [B, 2, T, F] <相应频率下的分量幅度在 实轴和虚轴的投影>
                                             dim=1)
                    batch_label = torch.stack(
                        (batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                        dim=1)
                batch_frame_num_list = batch.frame_num_list
                est_list = self.model(batch_feat)   # x_hat = model(y) [B, 2, T, F]

                batch_loss = eval(self.config.train.loss)(est_list, batch_label, batch_frame_num_list)  # loss class: mse...
                batch_loss.backward()
                self.optimizer.step()
                wandb.log(
                    {'train_batch_mse_loss': batch_loss.item()}
                )

            '''evaluate'''
            self.model.eval()
            all_loss_list = []
            all_csig_list, all_cbak_list, all_covl_list, all_pesq_list, all_ssnr_list, all_stoi_list = [], [], [], [], [], []
            with torch.no_grad():
                for batch in tqdm(self.cv_dataloader):
                    batch_feat = batch.feats.cuda()
                    batch_label = batch.labels.cuda()
                    noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])  # [B, 1, T, F]
                    clean_phase = torch.atan2(batch_label[:, -1, :, :], batch_label[:, 0, :, :])

                    '''four approaches for feature compression'''
                    if self.config.train.feat_type == 'normal':
                        batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
                    elif self.config.train.feat_type == 'sqrt':
                        batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                            torch.norm(batch_label, dim=1)) ** 0.5
                    elif self.config.train.feat_type == 'cubic':
                        batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                            torch.norm(batch_label, dim=1)) ** 0.3
                    elif self.config.train.feat_type == 'log_1x':
                        batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                                  torch.log(torch.norm(batch_label, dim=1) + 1)
                    if self.config.train.feat_type in ['normal', 'sqrt', 'cubic', 'log_1x']:
                        batch_feat = torch.stack(
                            (batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)), # [B, 2, T, F]
                            dim=1)
                        batch_label = torch.stack(
                            (batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                            dim=1)

                    est_list = self.model(batch_feat)   # [B, 2, T, F]
                    # est_list = batch_feat

                    batch_loss = eval(self.config.train.loss)(est_list, batch_label, batch.frame_num_list)
                    # print("evaluate loss: ", batch_loss)
                    batch_result = compare_complex(est_list, batch_label, batch.frame_num_list,
                                                   feat_type=self.config.train.feat_type)   # compute evaluate metrics
                    all_loss_list.append(batch_loss.item())
                    all_csig_list.append(batch_result[0])
                    all_cbak_list.append(batch_result[1])
                    all_covl_list.append(batch_result[2])
                    all_pesq_list.append(batch_result[3])
                    all_ssnr_list.append(batch_result[4])
                    all_stoi_list.append(batch_result[5])

                wandb.log(
                    {
                        'test_mean_mse_loss': np.mean(all_loss_list),   # mean loss in val dataset
                        'test_mean_csig': np.mean(all_csig_list),
                        'test_mean_cbak': np.mean(all_cbak_list),
                        'test_mean_covl': np.mean(all_covl_list),
                        'test_mean_pesq': np.mean(all_pesq_list),
                        'test_mean_ssnr': np.mean(all_ssnr_list),
                        'test_mean_stoi': np.mean(all_stoi_list),
                    }
                )

                cur_avg_loss = np.mean(all_loss_list)

                '''Adjust the learning rate and early stop'''
                if self.config.optim.half_lr > 1:
                    if cur_avg_loss >= prev_cv_loss:
                        cv_no_impv += 1
                        if cv_no_impv == self.config.optim.half_lr: # adjust lr depend on cv_no_impv
                            harving = True
                        if cv_no_impv >= self.config.optim.early_stop > 0:  # early stop
                            logging.info("No improvement and apply early stop")
                            break
                    else:
                        cv_no_impv = 0

                if harving == True:
                    optim_state = self.optimizer.state_dict()
                    for i in range(len(optim_state['param_groups'])):
                        optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                    self.optimizer.load_state_dict(optim_state)
                    logging.info('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                    harving = False
                prev_cv_loss = cur_avg_loss

                if cur_avg_loss < best_cv_loss:
                    logging.info(
                        f"last best loss is: {best_cv_loss}, current loss is: {cur_avg_loss}, save best_checkpoint.pth")
                    best_cv_loss = cur_avg_loss
                    states = [
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))
            # save latest checkpoint
            states = [
                self.model.state_dict(),
                self.optimizer.state_dict(),
            ]
            torch.save(states, os.path.join(self.args.checkpoint, f'checkpoint_{epoch}.pth'))

    def generate_wav(self, load_pre_train=True, data_path='data/noisy_testset_wav'):
        if load_pre_train:
            # load pretrained_model
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'checkpoint_49.pth'))
            self.model.load_state_dict(pretrained_data[0])
        self.model.eval()
        data_paths = glob.glob(data_path + '/*.wav')
        '''generate wav'''
        with torch.no_grad():
            for path in tqdm(data_paths):
                feat_wav, _ = librosa.load(path, sr=16000)
                c = np.sqrt(np.sum((feat_wav ** 2)) / len(feat_wav))
                feat_wav = feat_wav / c
                feat_wav = torch.FloatTensor(feat_wav)
                '''这里没有像 train 的时候 进行补零(collate.collate_fn) 虽然不会对输入model的数据维数产生影响，会不会对 wav_len < chunk_length 的样本产生影响'''
                feat_x = torch.stft(feat_wav,
                                    n_fft=self.config.train.fft_num,
                                    hop_length=self.config.train.win_shift,
                                    win_length=self.config.train.win_size,
                                    window=torch.hann_window(self.config.train.fft_num)).permute(2, 1, 0).cuda()
                feat_phase_x = torch.atan2(feat_x[-1, :, :], feat_x[0, :, :])
                if self.config.train.feat_type == 'sqrt':
                    feat_mag_x = torch.norm(feat_x, dim=0)
                    feat_mag_x = feat_mag_x ** 0.5
                feat_x = torch.stack(
                    (feat_mag_x * torch.cos(feat_phase_x), feat_mag_x * torch.sin(feat_phase_x)),
                    dim=0)  # [2, T, F]
                esti_x = self.model(feat_x.unsqueeze(dim=0)).squeeze(dim=0)
                esti_mag, esti_phase = torch.norm(esti_x, dim=0), torch.atan2(esti_x[-1, :, :],
                                                                              esti_x[0, :, :])
                if self.config.train.feat_type == 'sqrt':
                    esti_mag = esti_mag ** 2
                    esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=0)
                tf_esti = esti_com.permute(2, 1, 0).cpu()
                t_esti = torch.istft(tf_esti,
                                     n_fft=self.config.train.fft_num,
                                     hop_length=self.config.train.win_shift,
                                     win_length=self.config.train.win_size,
                                     window=torch.hann_window(self.config.train.fft_num),
                                     length=len(feat_wav)).numpy()
                t_esti = t_esti * c
                raw_path = path.split('/')[-1]
                sf.write(os.path.join(self.args.generated_wav, raw_path), t_esti, 16000)
        clean_data_path = 'data/clean_testset_wav'
        res = compare(clean_data_path, self.args.generated_wav)
        # res = compare(clean_data_path, data_path)
        pm = np.array([x[0:] for x in res])
        pm = np.mean(pm, axis=0)
        logging.info(f'ref={clean_data_path}')
        logging.info(f'deg={self.args.generated_wav}')
        logging.info('csig:%6.4f cbak:%6.4f covl:%6.4f pesq:%6.4f ssnr:%6.4f stoi:%6.4f' % tuple(pm))
