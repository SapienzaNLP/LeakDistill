import torch
from torch import nn
from torch.nn import functional as F
from geomloss import SamplesLoss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

class DistilLoss(nn.Module):
    def __init__(self, temp=1, embed_dim=1024, num_bins=3000, train_random_cls=True, encoder_layer_to_learn=12, lm_head=None):
        super().__init__()
        self.temp = temp
        self.encoder_layer_to_learn = encoder_layer_to_learn

        self.softmax = nn.Softmax(dim=-1)
        self.mse = nn.MSELoss()
        self.kd = nn.KLDivLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss()
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.random_cls = nn.Sequential(
            nn.Linear(embed_dim, num_bins)
        )
        if not train_random_cls:
            for p in self.random_cls.parameters():
                p.requires_grad_(False)

        self.was_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.triple_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.huber_loss = nn.HuberLoss()
        self.eps = 0.00000001

        self.rkd_dist = RkdDistance()
        self.rkd_a_dist = RKdAngle()

        if lm_head is not None:
            bart_vocab_size = 50265
            new_lm_size = lm_head.weight.shape[0] - bart_vocab_size
            self.lm_head = nn.Linear(lm_head.weight.shape[1], new_lm_size, bias=False)
            with torch.no_grad():
                self.lm_head.weight[:, :] = lm_head.weight[bart_vocab_size:]
        else:
            self.lm_head = self.random_cls

    def calc_kl_div(self, stud_logits, teach_logits):
        soft_pred = F.log_softmax(stud_logits / self.temp, dim=-1)
        soft_targets = self.softmax(teach_logits / self.temp).detach()
        return self.kd(soft_pred, soft_targets).sum(dim=-1).mean()

    def RKD(self, teacher_states, student_states):
        return self.rkd_dist(student_states, teacher_states)

    def RKD_A(self, teacher_states, student_states):
        pass

    def forward(self, metrics, teacher_output, student_output, target_mask, encoder_mask, model=None):
        loss = 0
        layer_to_learn = self.encoder_layer_to_learn
        for metric in metrics:
            if metric == 'kl':
                distil_loss = self.temp**2 * self.calc_kl_div(
                    student_output.logits[target_mask],
                    teacher_output.logits[target_mask],
                )
                loss += distil_loss
            elif metric == 'mse':
                mse_loss = 10*self.mse(
                    student_output.encoder_hidden_states[layer_to_learn][encoder_mask],
                    teacher_output.encoder_hidden_states[layer_to_learn][encoder_mask].detach()
                )
                loss += mse_loss
            elif metric == 'attention':
                attention_loss = 0
                log_eps = 0.00000001
                batch_size, head_size, seq_size, _ = student_output.encoder_attentions[0].shape
                attention_loss += self.kd(torch.log(student_output.encoder_attentions[layer_to_learn - 1] + log_eps),
                                          teacher_output.encoder_attentions[layer_to_learn - 1].detach()).flatten().sum() / batch_size / head_size / seq_size
                loss += attention_loss
            elif metric == 'cos':
                cosine_loss = self.cosine_sim(
                    teacher_output.encoder_hidden_states[layer_to_learn][encoder_mask].detach(),
                    student_output.encoder_hidden_states[layer_to_learn][encoder_mask])
                cosine_loss = (1 - torch.mean(cosine_loss)) / 2
                loss += cosine_loss
            elif metric == 'lm_fake_kl':
                lm_head = self.lm_head
                fake_cls_loss = self.calc_kl_div(
                    lm_head(student_output.encoder_last_hidden_state[encoder_mask]),
                    lm_head(teacher_output.encoder_last_hidden_state[encoder_mask]),
                )
                loss += fake_cls_loss
            else:
                raise 'wrong metrtic name'

        return loss

