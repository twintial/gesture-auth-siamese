def print_log(cost_time, train_loss, train_acc, val_loss, val_acc):
    log_str = f'- {cost_time:.0f}s - loss:{train_loss:.4f} - acc:{train_acc:.4f} ' \
        f'- val_loss:{val_loss:.4f} - val_acc:{val_acc:.4f}'
    print(log_str)


# 通用
def print_status_bar_ver0(time, *metrics, **kwargs):
    log_str = f'- {time:.0f}s'
    for m in metrics:
        log_str += f' - {m.name}:{m.result():.4f}'
    for k, v in kwargs.items():
        log_str += f' - {k}:{v:.4f}'
    print(log_str)


def print_status_bar_ver1(time, loss):
    log_str = f'- {time:.0f}s - {loss.name}:{loss.result():.4f}'
    print(log_str)
