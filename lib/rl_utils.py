import torch


def get_reward(logits, gt):
    """Per-step reward when the scaling-factor head is on.

    Hardcoded to the `corr+suc` combination that was the only velocity
    option we ran (was option='internal_corr_suc_3' before the refactor):
      - base reward = -CE(logits, gt) (detached)
      - corr filter: zero out the reward at steps where the argmax
        already matches gt (only mistakes get credited)
      - suc bonus: +100 on the last step if every step's argmax matched gt

    Args:
        logits: (T, num_class) per-step logits.
        gt:     (T,) per-step ground-truth action indices.

    Returns:
        reward: (T,) per-step reward tensor (no gradient).
    """
    reward = -torch.nn.CrossEntropyLoss(reduction='none')(logits, gt).detach()
    actions = logits.argmax(dim=-1)
    incorrect = (actions != gt).float()
    reward = reward * incorrect
    if torch.all(actions == gt):
        reward[-1] = 100
    return reward


def reinforce(prob, reward, gamma=0.9):
    """REINFORCE policy loss using the discounted-return reward shape.

    Hardcoded to the no-`gae`, no-`instant`, no-value-function path
    (was option='internal_corr_suc_3' which contained none of those).

    Args:
        prob:   (T, num_class) softmax-normalised action probabilities.
        reward: (T,) per-step reward (see get_reward).
        gamma:  discount factor.

    Returns:
        Scalar policy loss = -sum( returns * log_prob_of_chosen_action ).
    """
    actions = prob.argmax(dim=-1)
    log_prob = prob.log()
    mask = torch.zeros_like(log_prob)
    mask[torch.arange(log_prob.size(0)), actions] = 1
    log_prob = (log_prob * mask).sum(-1)
    returns = get_reinforce_return(reward, gamma)
    policy_loss = -returns * log_prob
    return policy_loss.sum()


def get_reinforce_return(reward, gamma):
    """Discounted-return sequence for REINFORCE.

    Args:
        reward: (T,) per-step reward.
        gamma:  discount factor.

    Returns:
        returns: (T,) where returns[t] = sum_{k>=t} gamma^(k-t) * reward[k].
    """
    R = 0
    returns = []
    for i in reversed(range(len(reward))):
        R = reward[i:i+1] + gamma * R
        returns = [R] + returns
    returns = torch.cat(returns)
    return returns
