import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from utils import *
from model import *
import logging
import wandb
from tqdm import tqdm

wandb.init(project="ddpm")


class ComplexDDPMTrainer(object):
    def __init__(self, args, config):
        # config
        self.args = deepcopy(args)
        self.config = deepcopy(config)
        '''dataset & dataloader'''
        collate = Collate(self.config)
        tr_dataset = VBTrDataset('data/noisy_trainset_wav', 'data/clean_trainset_wav', config)
        cv_dataset = VBCvDataset('data/noisy_testset_wav', 'data/clean_testset_wav', config)
        logging.info(f'Total {tr_dataset.__len__()} train data.')  # 11572
        logging.info(f'Total {cv_dataset.__len__()} eval data.')  # 824
        self.tr_dataloader = DataLoader(tr_dataset,
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
        '''optimizer'''
        if self.config.optim.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                self.config.optim.lr,
                weight_decay=self.config.optim.l2
            )

        if self.args.retrain:
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))
            self.model.load_state_dict(pretrained_data[0])
            self.optimizer.load_state_dict(pretrained_data[1])

        '''logger'''
        wandb.watch(self.model, log="all")

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
                noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])
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
                    batch_feat = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                                             dim=1)
                    batch_label = torch.stack(
                        (batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                        dim=1)
                batch_frame_num_list = batch.frame_num_list
                est_list = self.model(batch_feat)

                batch_loss = eval(self.config.train.loss)(est_list, batch_label, batch_frame_num_list)
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
                    noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])
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
                            (batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                            dim=1)
                        batch_label = torch.stack(
                            (batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                            dim=1)

                    est_list = self.model(batch_feat)
                    # est_list = batch_feat

                    batch_loss = eval(self.config.train.loss)(est_list, batch_label, batch.frame_num_list)
                    batch_result = compare_complex(est_list, batch_label, batch.frame_num_list,
                                                   feat_type=self.config.train.feat_type)
                    all_loss_list.append(batch_loss.item())
                    all_csig_list.append(batch_result[0])
                    all_cbak_list.append(batch_result[1])
                    all_covl_list.append(batch_result[2])
                    all_pesq_list.append(batch_result[3])
                    all_ssnr_list.append(batch_result[4])
                    all_stoi_list.append(batch_result[5])

                wandb.log(
                    {
                        'test_mean_mse_loss': np.mean(all_loss_list),
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
                        if cv_no_impv == self.config.optim.half_lr:
                            harving = True
                        if cv_no_impv >= self.config.optim.early_stop > 0:
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
                        f"best loss is: {best_cv_loss}, current loss is: {cur_avg_loss}, save best_checkpoint.pth")
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
                    dim=0)
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
