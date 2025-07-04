import torch
import torch.nn as nn

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)

class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)

class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss

class BloLoss(nn.Module):
    def __init__(self, alpha):
        super(BloLoss,self).__init__()
        self.alpha = alpha
    def forward(self, log_output, target, label):
        class_num = log_output.shape[1]

        log_prob_right = log_output.t()[torch.arange(log_output.size(1)) != 0].t()  ## bs * (class_num - 1)
        log_prob_left = log_output.t()[torch.arange(log_output.size(1)) != 40].t()  ## bs * (class_num - 1)

        label_left = torch.unsqueeze(label[torch.arange(label.size(0)) != 40], 0)  ## 1 * (class_num - 1)
        sign_right = torch.ge(label_left, target).float() ## bs * (class_num - 1)
        sign_left = torch.lt(label_left, target).float()  ## bs * (class_num - 1)

        ## BLO
        div_value = 1 / torch.tensor(class_num - 1)
        relu_loss = torch.nn.ReLU()
        margin_right = torch.tensor(1) - relu_loss(label_left - target)
        margin_left = torch.tensor(1) - relu_loss(target - label_left - torch.tensor(div_value))
        margin_left *= self.alpha
        margin_right *= self.alpha

        loss = sign_right * relu_loss(log_prob_right - log_prob_left + margin_right) \
               + sign_left * relu_loss(log_prob_left - log_prob_right + margin_left)

        return torch.mean(torch.sum(loss, dim=-1))