import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from datetime import datetime
import sys
from scipy.signal import resample
from sklearn.metrics import accuracy_score
import argparse
from eegnet import EEGNet
from config import Config, rayConfig2a
from FeatureSelector.mars import mars_selector_oneshot as mars_selector
# from train_agent import mars_agent_gradient
from model_agent import ACTOR, CRITIC
# from torch.utils.tensorboard import SummaryWriter
# import wandb
# os.environ["WANDB_API_KEY"] = "YOUR WANDB API KEY"

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers.pb2 import PB2
from seed import set_seed
args = rayConfig2a()
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
os.environ["CUDA_VISIBLE_DEVICES"] = f"2,3,4,5"
os.environ["WANDB_MODE"] = "disabled"

# pretrained model timestamp
timestamp_list = ["220530.113310", "220530.114703", "220530.114628",
                  "220530.115302", "220530.115345", "220530.114808",
                  "220530.120415", "220530.120704", "220530.121420"]


class experiment():
    def __init__(self, train_X, train_Y, test_X, test_Y):
        # Load dataset.
        self.Xtr, self.Ytr = torch.FloatTensor(train_X).cuda(), torch.LongTensor(train_Y).cuda()
        self.Xts, self.Yts = torch.FloatTensor(test_X).cuda(), torch.LongTensor(test_Y).cuda()

        self.init_LR = args.lr
        self.num_epochs_pre = args.num_epochs_pre  # Pre-training epochs
        self.num_epochs = args.num_epochs
        self.num_batch = args.batch_size
        self.weight_decay = args.weight_decay
        self.AC_LR = args.AClr
        self.AC_WD = args.ACwd
        self.sbj_idx = args.subjectID
        print(f"START TRAINING Subject {self.sbj_idx}")
        # Call optimizer.
        self.num_batch_iter = int(self.Xtr.shape[0] / self.num_batch)
        self.criterion = nn.CrossEntropyLoss(reduction='none')


    def training_AM(self, config, checkpoint_dir=None):
        # os.environ["WANDB_MODE"] = "offline"
        # if iteration == num_iter:
        #     wandb.init(project="MARS", entity="yourID", tags=[f"Subject {args.subjectID}", "end"], reinit=True)
        # else:
        # wandb.init(project="MARS", entity="yourID", tags=[f"Subject {args.subjectID}"], reinit=True)
        # wandb.config.update(args)
        # wandb.run.log_code(".")

        gam = config["gam"]
        r_gam = config["r_gam"]
        self.AC_LR = config["AClr"]
        self.AC_WD = config["ACwd"]

        timestamp = timestamp_list[args.subjectID - 1]
        save_dir = f"/PRETRAIN_DIR/subject{self.sbj_idx}/"

        os.makedirs(save_dir, exist_ok=True)
        PATH = save_dir + f"{timestamp}.pt"
        EEG = EEGNet(args=None, shape=[20, 1, 22, 1200])
        EEG.load_state_dict(torch.load(PATH, map_location=self.Xts.device))
        EEG = EEG.cuda()
        EEG_optimizer = torch.optim.RMSprop(params=EEG.parameters(), lr=self.init_LR, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(EEG_optimizer, gamma=0.999)

        # To record the loss curve.
        loss_AM = []

        print(f"Subject {self.sbj_idx}")
        actor_temporal = ACTOR().cuda()
        critic_centralized = CRITIC().cuda()
        actor_spectral = ACTOR().cuda()

        actor_temporal_optimizer = torch.optim.RMSprop(params=actor_temporal.parameters(), lr=self.AC_LR, weight_decay=self.AC_WD)
        critic_centralized_optimizer = torch.optim.RMSprop(params=critic_centralized.parameters(), lr=self.AC_LR, weight_decay=self.AC_WD)

        actor_spectral_optimizer = torch.optim.RMSprop(params=actor_spectral.parameters(), lr=self.AC_LR, weight_decay=self.AC_WD)
        epoch = 0
        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(path)
            EEG.load_state_dict(checkpoint["EEG_model_state_dict"])
            actor_spectral.load_state_dict(checkpoint["actor_spec_model_state_dict"])
            actor_temporal.load_state_dict(checkpoint["actor_temp_model_state_dict"])
            critic_centralized.load_state_dict(checkpoint["critic_cent_model_state_dict"])
            epoch = checkpoint["step"]

        while True:
            EEG.train()
            rand_idx = np.random.permutation(self.Xtr.shape[0])
            self.Xtr, self.Ytr = self.Xtr[rand_idx, :, :, :], self.Ytr[rand_idx, :]
            loss_per_epoch = 0
            start_time = time.time()
            Ytr_hat = []
            for batch in range(self.num_batch_iter):
                EEG_optimizer.zero_grad()
                # Sample minibatch.
                xb = self.Xtr[batch * self.num_batch: (batch + 1) * self.num_batch, :, :, :]
                yb = self.Ytr[batch * self.num_batch: (batch + 1) * self.num_batch, :]
                yb = torch.argmax(yb, dim=-1)
                # Extract full segments.
                ################################################
                out = EEG.temporal_conv(xb)  # B, 16, 22, 1125
                out = EEG.spatial_conv(out)  # B, 32, 1, 281
                out = EEG.separable_conv(out)  # B, 32, 1, 33

                mask_mars = mars_selector(out.clone().detach(), self.num_batch, actor_temporal, actor_spectral,
                                          critic_centralized, actor_temporal_optimizer, actor_spectral_optimizer,
                                          critic_centralized_optimizer,
                                          EEG, xb, yb, self.criterion, epoch, gam=gam, r_gam=r_gam)
                out = out * mask_mars.clone().detach()

                ################################################

                # wandb.log({
                #     f"Train Temp mask ratio": mask_mars.sum() / mask_mars.numel(),
                #     },
                #     step=epoch)

                pred_out = EEG.classifier(out)
                loss = self.criterion(pred_out, yb).mean()

                loss.backward()
                EEG_optimizer.step()

                loss_AM.append(loss.item())
                loss_per_epoch += loss.item()
                Ytr_hat.extend(np.argmax(pred_out.clone().detach().cpu().numpy(), axis=-1))
            loss_per_epoch /= self.num_batch_iter
            train_loss = loss_per_epoch
            Ytr_hat = np.array(Ytr_hat)
            # train_acc = accuracy_score(np.argmax(self.Ytr.clone().detach().cpu().numpy(), axis=-1), Ytr_hat)
            # wandb.log({f"Train Score": train_acc.item(),
            #            f"Train Loss": loss_per_epoch,
            #            },
            #           step=epoch)
            scheduler.step()

            Yts_hat = []
            num_test_batch = int(self.Xts.shape[0] / self.num_batch)
            with torch.no_grad():
                EEG.eval()
                loss_per_epoch = 0
                for batch in range(num_test_batch):
                    xb = self.Xts[batch * self.num_batch: (batch + 1) * self.num_batch, :, :, :]
                    yb = self.Yts[batch * self.num_batch: (batch + 1) * self.num_batch, :]
                    yb = torch.argmax(yb, dim=-1)
                    # pred_out = EEG(xb).clone().detach().cpu()
                    # Yts_hat.extend(np.argmax(pred_out.numpy(), axis=-1))
                    out = EEG.temporal_conv(xb)  # B, 16, 22, 1125
                    out = EEG.spatial_conv(out)  # B, 32, 1, 281
                    out = EEG.separable_conv(out)  # B, 32, 1, 33

                    # out = gru(out)

                    mask_mars = mars_selector(out.clone().detach(), self.num_batch, actor_temporal, actor_spectral,
                                              critic_centralized, actor_temporal_optimizer, actor_spectral_optimizer,
                                              critic_centralized_optimizer,
                                              EEG, xb, yb, self.criterion, epoch, train_type='test')

                    out = out * mask_mars.clone().detach()

                    # wandb.log({
                    #     f"Test Temp mask ratio": mask_mars.sum() / mask_mars.numel(),
                    #     },
                    #     step=epoch)

                    pred_out = EEG.classifier(out)
                    loss = self.criterion(pred_out, yb).mean()
                    loss_per_epoch += loss.item()
                    Yts_hat.extend(np.argmax(pred_out.clone().detach().cpu().numpy(), axis=-1))
                loss_per_epoch /= self.num_batch_iter
                Yts_hat = np.array(Yts_hat)
                test_acc = accuracy_score(np.argmax(self.Yts.clone().detach().cpu().numpy(), axis=-1), Yts_hat)
            # wandb.log({f"Test Score": test_acc.item(),
            #            f"Test Loss": loss_per_epoch,
            #            },
            #           step=epoch)

            if epoch % 36 == 0:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    # Then create a checkpoint file in this directory.
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    # Save state to checkpoint file.
                    # No need to save optimizer for SGD.

                    torch.save({
                        "step": epoch,
                        "EEG_model_state_dict": EEG.state_dict(),
                        "actor_spec_model_state_dict": actor_spectral.state_dict(),
                        "actor_temp_model_state_dict": actor_temporal.state_dict(),
                        "critic_cent_model_state_dict": critic_centralized.state_dict(),
                        "mean_accuracy": test_acc,
                        "loss": train_loss
                    }, path)
            epoch += 1
            tune.report(accuracy=test_acc, loss=train_loss)


def main(num_samples=10, gpus_per_trial=1):

    perturbation_interval = 36
    config = {
        "r_gam": tune.loguniform(1e-1, 1e+1),
        "gam": tune.uniform(0.9, 0.99),
        "AClr": tune.loguniform(1e-05, 1e-03),
        "ACwd": tune.loguniform(1e-05, 1e-03),
        # 'wandb': {
        #     'project': "MARS",
        #     'api_key': "YOUR API KEY",
        #     'log_config': True,
        #     'entity': "YOURID",
        #     'tags': [f"Subject {args.subjectID}"]
        # }
    }

    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     grace_period=1,
    #     reduction_factor=2)
    scheduler = PB2(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_bounds={
            # distribution for resampling
            "r_gam": [0.05, 1.0],
            "gam": [0.9, 0.99],
            "AClr": [1e-05, 1e-03],
            "ACwd": [1e-05, 1e-03],
            # allow perturbations within this set of categorical values
            # "momentum": [0.8, 0.9, 0.99],
        })
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["accuracy", "loss"])
    result = tune.run(
        partial(loveray),
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        metric="loss",  # tune_report로 리포트 해야함.
        mode="min",
        stop={
            "accuracy": 0.99,  # subject2 = 72
            "training_iteration": args.num_epochs - args.num_epochs_pre,
        },
        reuse_actors=True,
        checkpoint_score_attr="loss",
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(
    #     best_trial.last_result["loss"]))
    print("Best trial final test accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final train loss: {}".format(
        best_trial.last_result["loss"]))

    # save_dir = "Your directory"
    # path = save_dir + f"/subject{args.subjectID}.csv"
    # a = pd.DataFrame({})
    # # a = a.append({"acc": best_trial.last_result["accuracy"], "config": best_trial.config}, ignore_index=True)
    # a = a.append({"acc": best_trial.last_result["accuracy"], "config": best_trial.config, "seed": args.seed}, ignore_index=True)
    #
    # # os.makedirs(path, exist_ok=True)
    # if os.path.exists(path):
    #     a.to_csv(f"{path}", mode='a', header=False)
    # else:
    #     a.to_csv(f"{path}", mode='w')


def loveray(config, checkpoint_dir=None):
    seed_iter = 1
    subj = args.subjectID
    data_dir = "DATA_DIR"
    # train_X = torch.load(f"{data_dir}/ku-mi/S{subj}_X_train.pt").numpy()
    # trY = torch.load(f"{data_dir}/ku-mi/S{subj}_y_train.pt").numpy()
    # test_X = torch.load(f"{data_dir}/ku-mi/S{subj}_X_test.pt").numpy()
    # teY = torch.load(f"{data_dir}/ku-mi/S{subj}_y_test.pt").numpy()
    train_X = np.load(f"{data_dir}/bcic_dataset/S0{subj}_train_X.npy")
    trY = np.load(f"{data_dir}/bcic_dataset/S0{subj}_train_y.npy")
    test_X = np.load(f"{data_dir}/bcic_dataset/S0{subj}_test_X.npy")
    teY = np.load(f"{data_dir}/bcic_dataset/S0{subj}_test_y.npy")

    train_Y = np.zeros((trY.size, trY.max() + 1))
    train_Y[np.arange(trY.size), trY] = 1
    test_Y = np.zeros((teY.size, teY.max() + 1))
    test_Y[np.arange(teY.size), teY] = 1
    exp = experiment(train_X, train_Y, test_X, test_Y)
    acc_mean = 0
    train_mean = 0
    for i in range(seed_iter):
        exp.training_AM(config, checkpoint_dir)
        # acc_mean += acc_AM
        # train_mean += train_loss
    # acc_mean /= seed_iter
    # train_mean /= seed_iter
    # tune.report(accuracy=acc_mean, loss=train_mean)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    set_seed(args.seed)
    main(num_samples=12, gpus_per_trial=1)

