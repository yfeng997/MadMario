import os

def clear_metrics():
    metrics = [
        'ep_reward',
        'ep_length',
        'ep_total_loss',
        'ep_total_q',
        'ep_learn_length',
        'mean_loss',
        'mean_q'
    ]
    return dict(zip(metrics, [0]*len(metrics)))


def collect_metrics(metrics, reward, loss, q):
    metrics['ep_reward'] += reward
    metrics['ep_length'] += 1
    metrics['mean_reward'] = round(metrics['ep_reward'] / metrics['ep_length'], 3)
    if loss is not None:
        metrics['ep_total_loss'] += loss
        metrics['ep_total_q'] += q
        metrics['ep_learn_length'] += 1
        metrics['mean_loss'] = round(metrics['ep_total_loss'] / metrics['ep_learn_length'], 3)
        metrics['mean_q'] = round(metrics['ep_total_q'] / metrics['ep_learn_length'], 3)
    return metrics

def log_metrics(metrics, log_dir):
    log_path = os.path.join(log_dir, "log.txt")
    metrics_str = display_metrics(metrics)
    with open(log_path, "a") as f:
        f.write(metrics_str + "\n")


def display_metrics(metrics):
    prefix = "[L]" if metrics['ep_learn_length']>0 else "[B]"
    return f"{prefix} | " \
        f"Episode length: {metrics['ep_length']} | " \
        f"Mean reward: {metrics['mean_reward']} | " \
        f"Mean loss: {metrics['mean_loss']} | " \
        f"Mean Q: {metrics['mean_q']} "
