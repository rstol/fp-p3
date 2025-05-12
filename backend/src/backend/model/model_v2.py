import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions.laplace import Laplace
from torch.optim.lr_scheduler import OneCycleLR

from backend.model.decoder import Decoder
from backend.model.encoder import Encoder

# COLORS = [(0,0,255), (255,0,255), (180,180,0), (143,143,188), (0,100,0), (128,128,0)]
# TrajCOLORS = [(0,0,255), (200,0,0), (200,200,0), (0,200,0)]


class PlayTransformer(pl.LightningModule):
    def __init__(self, in_feat_dim, time_steps, feature_dim, head_num, k, F, halfwidth, lr):
        super(PlayTransformer, self).__init__()
        # self.cfg = cfg
        self.in_feat_dim = in_feat_dim
        self.time_steps = time_steps
        # self.current_step = cfg.model.current_step
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k
        self.F = F
        self.halfwidth = halfwidth
        self.target_scale = 1.0
        self.lr = lr

        self.Loss = nn.MSELoss(reduction="none")

        self.encoder = Encoder(
            self.device, self.in_feat_dim, self.time_steps, self.feature_dim, self.head_num
        )
        self.decoder = Decoder(
            self.device, self.time_steps, self.feature_dim, self.head_num, self.k, self.F
        )

    def forward(
        self,
        states_batch,
        agents_batch_mask,
        states_padding_mask_batch,
        states_hidden_mask_batch,
        agent_ids_batch,
    ):
        e = self.encoder(
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_mask_batch,
            agent_ids_batch,
        )
        encodings = e["out"]
        # states_padding_mask_batch = states_padding_mask_batch + ~states_hidden_mask_batch
        decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch)

        return {
            "prediction": decoding.permute(1, 2, 0, 3),
            "att_weights": e["att_weights"],
        }  # [F,A,T,4] converted to [A,T,F,4]

    def get_encoder_output(
        self,
        states_batch,
        agents_batch_mask,
        states_padding_mask_batch,
        states_hidden_mask_batch,
        agent_ids_batch,
    ):
        e = self.encoder(
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_mask_batch,
            agent_ids_batch,
        )
        return e

    def training_step(self, batch, batch_idx):
        (
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_BP_batch,
            num_agents_accum,
            agent_ids_batch,
        ) = batch
        states_hidden_mask_batch = states_hidden_BP_batch
        print(torch.mean(states_hidden_mask_batch.float()).item())

        # Predict
        out = self(
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_mask_batch,
            agent_ids_batch,
        )
        prediction = out["prediction"]  # [A,T,F,4]

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch * states_hidden_mask_batch

        gt = states_batch[:, :, :2]  # A,T,2
        gt = gt.unsqueeze(2).repeat(1, 1, self.F, 1)  # [A,T,2] -> [A,T,F,2]
        gt_scale = torch.ones_like(gt).fill_(self.target_scale)
        gt_dist = Laplace(gt, gt_scale)  # [A,T,F,2]

        predict_dist = Laplace(prediction[:, :, :, 0::2], prediction[:, :, :, 1::2])  # [A,T,F,2]
        loss = torch.distributions.kl.kl_divergence(gt_dist, predict_dist)  # [A,T,F,2]

        loss_mask = (
            to_predict_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.F, 2)
        )  # [A,T] -> [A,T,F,2]

        loss_ = loss * loss_mask  # [A,T,F,2]
        loss_ = torch.sum(loss_, dim=3)  # [A,T,F]
        loss_ = torch.sum(loss_, dim=1)  # mean each trajectory [A,F]

        # per agent (so min across the future dimension). [A,F]
        marginal_loss = torch.min(loss_, dim=1).values  # [A]
        marginal_loss_ = torch.sum(marginal_loss)  # value

        # P_loss = loss_.view(-1, 6 , self.F) #[B,A,F]
        # joint_loss = torch.sum(P_loss,dim=1) #[B,F]
        joint_loss = torch.sum(loss_, dim=0)  # [F]
        joint_loss_ = torch.min(joint_loss)  #
        # joint_loss_ = torch.mean(joint_loss_)

        summary_loss = marginal_loss_ + 0.1 * joint_loss_

        """#MSE Loss
        loss_ = self.Loss(gt.unsqueeze(1).repeat(1,self.F,1), prediction)
        loss_ = torch.mean(torch.mean(loss_, dim=0),dim=-1) * self.halfwidth
        # loss_ = torch.min(loss_) 
        # if self.global_step < 8000:
        #     k_=4
        # elif self.global_step < 50000:
        #     k_ = 2
        # else:
        #     k_ = 1
        k_ = 1
        loss_, _ = torch.topk(loss_, k_)
        summary_loss = torch.mean(loss_)"""

        self.log_dict({"train/loss": summary_loss})

        self.logger.experiment.add_scalar("Loss/train", summary_loss.item(), self.global_step)
        print(summary_loss, "batchsize", states_batch.size())
        # return {'batch': batch, 'pred': prediction, 'gt': gt, 'loss': summary_loss 'att_weights': out['att_weights']}
        return summary_loss

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    break

        if not valid_gradients:
            print("detected inf or nan values in gradients. not updating model parameters")
            self.zero_grad()

    def validation_step(self, batch, batch_idx):
        (
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_BP_batch,
            num_agents_accum,
            agent_ids_batch,
        ) = batch

        states_hidden_mask_batch = states_hidden_BP_batch

        # Predict
        out = self(
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_mask_batch,
            agent_ids_batch,
        )
        prediction = out["prediction"]  # [A,T,F,4]

        # Calculate Loss
        to_predict_mask = (
            ~states_padding_mask_batch * states_hidden_mask_batch
        )  # [A,T] True for predict
        # repeat at feature dimension

        # to_predict_mask_gt = to_predict_mask.unsqueeze(2).repeat(1, 1, 2) #A,T,2
        # to_predict_mask_pre = to_predict_mask.unsqueeze(2),unsqueeze(3).repeat(1, self.F, 4)

        gt = states_batch[:, :, :2]  # A,T,2
        gt = gt.unsqueeze(2).repeat(1, 1, self.F, 1)  # [A,T,2] -> [A,T,F,2]
        gt_scale = torch.ones_like(gt).fill_(self.target_scale)
        gt_dist = Laplace(gt, gt_scale)  # [A,T,F,2]

        predict_dist = Laplace(prediction[:, :, :, 0::2], prediction[:, :, :, 1::2])  # [A,T,F,2]
        loss = torch.distributions.kl.kl_divergence(gt_dist, predict_dist)  # [A,T,F,2]

        loss_mask = (
            to_predict_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.F, 2)
        )  # [A,T] -> [A,T,F,2]

        loss_ = loss * loss_mask  # [A,T,F,2]
        loss_ = torch.sum(loss_, dim=3)  # [A,T,F]
        loss_ = torch.sum(loss_, dim=1)  # mean each trajectory [A,F]

        # per agent (so min across the future dimension). [A,F]
        marginal_loss = torch.min(loss_, dim=1).values  # [A]
        marginal_loss_ = torch.sum(marginal_loss)  # value

        # P_loss = loss_.view(-1, 6 , self.F) #[B,A,F]
        # joint_loss = torch.sum(P_loss,dim=1) #[B,F]
        joint_loss = torch.sum(loss_, dim=0)  # [F]
        joint_loss_ = torch.min(joint_loss)  #
        # joint_loss_ = torch.mean(joint_loss_)

        summary_loss = marginal_loss_ + 0.1 * joint_loss_

        rs_error = ((prediction[:, :, :, 0::2] - gt) ** 2).sum(dim=-1).sqrt_()  # [A,T,F]
        rs_error[~to_predict_mask] = 0  # [A,T,F]
        rse_sum = torch.sum(rs_error, dim=1)  # [A,F]
        ade_mask = to_predict_mask.sum(-1) != 0  # [A]
        ade = (rse_sum[ade_mask].permute(1, 0) / to_predict_mask[ade_mask].sum(-1)).permute(1, 0)

        fde_mask = to_predict_mask[:, -1] == True
        fde = rs_error[fde_mask][:, -1, :]

        minade, _ = ade.min(dim=-1)
        avgade = ade.mean(dim=-1)
        minfde, _ = fde.min(dim=-1)
        avgfde = fde.mean(dim=-1)

        batch_minade = minade.mean()
        batch_minfde = minfde.mean()
        batch_avgade = avgade.mean()
        batch_avgfde = avgfde.mean()

        self.log_dict(
            {
                "val/loss": summary_loss,
                "val/minade": batch_minade,
                "val/minfde": batch_minfde,
                "val/avgade": batch_avgade,
                "val/avgfde": batch_avgfde,
            }
        )

        self.val_out = {
            "states": states_batch,
            "states_padding": states_padding_mask_batch,
            "states_hidden": states_hidden_mask_batch,
            "num_agents_accum": num_agents_accum,
            "pred": prediction,
            "loss": summary_loss,
            "att_weights": out["att_weights"],
        }
        self.logger.experiment.add_scalar("Loss/valid", summary_loss, self.global_step)
        return summary_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))  # 1e-7
        scheduler = OneCycleLR(optimizer, max_lr=5e-7, steps_per_epoch=3039, epochs=50)
        lr = scheduler.get_last_lr()[0]
        print(lr)
        return [optimizer], [scheduler]
        # return optimizer
