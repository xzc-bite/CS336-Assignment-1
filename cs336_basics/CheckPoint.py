"""断点续训用的存取工具。

训练一次几小时到几天，中间机器重启、被抢占、或者只是想换个超参重跑，
都需要能从中断的地方接着来。而且"接着来"要真的等于"没断过"，
光存权重是不够的——优化器的状态必须一起存。

为什么优化器状态这么关键：
Adam 的 m（一阶动量）和 v（二阶矩）都是随训练累积的，还有 step 计数参与偏差修正。
只恢复 model.state_dict() 的话，m/v 归零、step 回到 0，效果是学习率突然被放大
（偏差修正把 lr 除以 1-β1^t，t 小的时候分母极小），loss 会先跳一下再慢慢爬回去。
这是续训最经典的一个坑，表现是"能跑，但曲线不像连续的"。

存什么：
    model_state_dict      全部参数；如果有 register_buffer（比如 RoPE 的位置表里
                          persistent=True 的那些）也会进 state_dict
    optimizer_state_dict  每个参数的 m、v、step，以及 param_groups 里的 lr/betas 等超参
    iteration             训练跑到第几步。lr schedule 靠它重算，日志横轴也靠它对齐

没存什么（续训时要自己在外面补）：
    - 随机数状态（torch / numpy / cuda）。不存的话数据采样顺序从头来过，
      严格 bit-level 复现做不到，但统计意义上一般可接受
    - 独立的 lr scheduler 状态。CosineSchedule 那种纯函数式调度器只要 iteration
      就能重算；如果用的是 nn.LRScheduler，它有自己的 state_dict()，得单独存一份
    - 梯度。不需要，load 之后重新 backward 就有了

复杂度：两个函数都是 O(N) 的 I/O，N 是参数量。大模型上 torch.save 会卡几秒到几十秒，
训练循环里别每一步都存，按步数间隔或按时间间隔来。
"""

import os

import torch


def save_checkpoint(model, optimizer, iteration, out):
    """把模型、优化器和步数打包写盘。

    参数:
        model: torch.nn.Module，要保存的模型。存的是 state_dict() 不是模型对象本身——
               存对象会把整个类定义和构造参数一起 pickle 进去，
               以后改了类名/文件位置就再也读不出来。
        optimizer: torch.optim.Optimizer，必须是用 model.parameters() 构造的那个，
               两者参数顺序一致才能保证恢复时状态对得上号。
        iteration: int，>= 0，已经完成的训练步数。
               注意语义是"完成了多少步"而不是"下一步是几"，
               续训时如果直接 for it in range(iteration, total) 就别再 +1。
        out: 保存路径（str / os.PathLike）或可写文件对象（BinaryIO，比如 io.BytesIO）。
               torch.save 两种都支持；传路径时父目录必须已存在，它不会自动建。

    返回: None。没有返回值是刻意的——写盘失败会直接抛异常，不需要靠返回值判断。

    维度说明：这里没有形状变换，只是把一堆张量按原样序列化。
              state_dict 是 dict[str, Tensor]，key 是参数名（'layers.0.attn.q_proj.weight'
              这类），value 保持参数原本的形状。文件总大小 ≈ 参数量 × 4 字节
              （fp32）+ 优化器状态的 2 倍，也就是参数量的 12 倍量级。
    """
    # state_dict() 返回的是引用（共享同一块内存），不是拷贝。
    # 这里安全，因为 torch.save 会立刻把内容写出去，不会留到后面异步执行。
    # 但如果哪天改成先构造好 dict、隔几步再存，就必须 clone，否则存下去的是"最新"的权重。
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    # 原子写：先落 .tmp，写完整了再 os.replace 换过去。
    # 直接覆盖原文件的话，写一半断电/被 kill，旧 checkpoint 也一起废了，
    # 长训练跑一晚上就白费。os.replace 在同一文件系统上是原子的，
    # 任何时刻磁盘上的 ckpt.pt 要么是旧的完整文件，要么是新的完整文件。
    #
    # 传文件对象（io.BytesIO）的调用方拿不到路径，就只能直接写，无法原子化——
    # 测试里用的就是这种，无所谓，内存对象不存在"写一半"。
    if isinstance(out, (str, os.PathLike)):
        tmp_path = f"{os.fspath(out)}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, os.fspath(out))
    else:
        torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """从 checkpoint 恢复模型和优化器，返回当初存的训练步数。

    参数:
        src: 读取路径（str / os.PathLike）或可读文件对象（BinaryIO，
             例如 seek(0) 之后的 io.BytesIO）。测试里用的就是后者。
        model: torch.nn.Module，结构必须和保存时一致。
               少一层、改个维度都会让 load_state_dict 在 strict 模式下报 key/shape 不匹配。
        optimizer: torch.optim.Optimizer，且必须已经绑定到 model.parameters()。
               空优化器（没绑参数）直接 load 会报 param_groups 长度对不上。

    返回:
        int，checkpoint 里记录的 iteration。调用方拿它去重算 lr、接上日志：
            lr = schedule(iteration)
            for it in range(iteration, total_iters): ...
    """
    # map_location="cpu" 不能省，两个方向都会出事：
    #   - GPU 上存的 ckpt 拿到只有 CPU 的机器上读：直接 RuntimeError
    #   - 多卡训练时，torch.load 默认把张量恢复到保存时所在的设备（通常是 cuda:0），
    #     于是所有卡的 checkpoint 内容都先挤到 0 号卡上，显存瞬间爆掉
    # 先统一落到内存，再由 load_state_dict 按目标参数所在设备分配，跨机器续训才安全。
    #
    # 另外 torch 2.6 起 weights_only 默认是 True（实测 2.11 确实如此）：
    # 只允许张量、基本容器和标量，遇到自定义类直接 UnpicklingError。
    # 本 checkpoint 里只有 dict/Tensor/int，正好合规。
    # 如果以后往里面塞 dataclass 之类，要么显式 weights_only=False，
    # 要么把它拆成 dict/基本类型再存——后者更安全，因为 weights_only=False
    # 等于允许任意代码执行，别拿它去读来路不明的文件。
    checkpoint = torch.load(src, map_location="cpu")

    # 用 load_state_dict 而不是 model.load_state_dict 之外的赋值方式：
    # 它会按键逐个 copy_ 到已有的 Parameter 上，保持 Parameter 对象不变。
    # 这一点对优化器至关重要——optimizer.state 是以 Parameter 对象为 key 的，
    # 把参数整个替换成新张量，优化器里那些 m/v 就全丢了。
    model.load_state_dict(checkpoint['model_state_dict'])

    # 这个恢复的是每个参数的 m/v/step（张量，形状与对应参数一致）
    # 以及 param_groups 里的超参。要求两侧 param_groups 的数量和每个组里的参数个数一致，
    # 否则报 "loaded state dict contains a parameter group that doesn't match"。
    # 所以分层学习率（多个 param_group）的模型，恢复时必须先按同样方式分组再 load。
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 纯 Python int，恢复后可以安全地参与 range()、写日志、算 schedule。
    iteration = checkpoint['iteration']
    return iteration