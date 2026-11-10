import time  # 计时用的标准库；time.time() 取的是"墙上时钟"的秒数，用来量一段代码到底跑了多久
import torch  # PyTorch 主库：这个脚本干的事就是用 GPU 跑一个最简单的线性层，然后比速度


# 这个脚本想回答一个很朴素的问题：同一段计算，"1 句一次算" 和 "64 句打包一起算"，哪个快？
# 答案有点反直觉——打包算 64 句的耗时，和单独算 1 句几乎一样。GPU 是"广撒网"型选手，
# 一次只喂 1 句的话，绝大部分算力都在空转。这也是为什么现在的训练和推理都拼命讲究"批处理"。

def main():
    # 换成 cpu 之后逻辑不变，只是慢，方便clone 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.nn.Linear(1024, 1024).to(device)  # 一个最简单的全连接层：y = xW^T + b，输入输出都是 1024 维
    # 模拟单句
    x_single = torch.randn(1, 1024, device=device)  # batch=1：1 行 × 1024 列
    # 模拟批处理
    x_batch = torch.randn(64, 1024, device=device)  # batch=64：64 行 × 1024 列；列数必须还是 1024，否则对不上模型

    # 第一次调用会做 cuBLAS 句柄初始化、显存分配等一次性开销，
    # 不预热的话这份开销全砸在"第一组"头上，单句那组会被明显高估，
    # 两边就不是在同一起跑线上比了。
    for _ in range(3):
        _ = model(x_single)
        _ = model(x_batch)
    if device == "cuda":
        torch.cuda.synchronize()  # GPU 是异步的：CPU 只负责"下单"就返回，得等它真算完

    # 测试单句耗时
    if device == "cuda":
        torch.cuda.synchronize()  # 起表前再同步一次，确保前面热身的活儿全部收尾
    start = time.time()  # 记下起始时间戳（单位是秒）
    for _ in range(100):  # 重复 100 次：冷启动已经被热身吃掉了，这里量的是稳态耗时
        _ = model(x_single)  # 真正跑一次前向；输出接到 _ 上，一是保证结果不被立刻回收，二是 _ 是"这个值我暂时用不到"的约定写法
    if device == "cuda":
        torch.cuda.synchronize()  # 停表前必须同步，否则量到的只是"CPU 下单用了多久"
    print(f"单句 100 次耗时: {time.time() - start:.4f}s")  # :.4f 保留 4 位小数，不然打印一长串看着乱

    # 测试批处理耗时
    start = time.time()  # 重新起表，这次量批量那一组
    for _ in range(100):  # 同样跑 100 次，保证和上面可比
        _ = model(x_batch)  # 一次前向就把 64 行全算完；GPU 内部会把 64×1024 的矩阵乘拆成大量小 kernel 并行执行
    if device == "cuda":
        torch.cuda.synchronize()  # 同样要等 GPU 真的算完再停表
    print(f"64 句 100 次耗时: {time.time() - start:.4f}s")


# main 保护：这样 import 这个文件时不会顺手跑一遍基准、占一块显存。
if __name__ == "__main__":
    main()


# 看完两行输出，试着算一算这几个比值：
#   · 总耗时比：批处理 / 单句 —— 越接近 1，说明 GPU 越是被单句喂不饱；
#   · 单句耗时比：总耗时 / 64 —— 批处理摊到每句上的真实成本，是最该关心的指标；
#   · 加速比：单句总耗时 / 批处理总耗时 —— 也就是"批处理"给你省下了多少倍的时间。
#
# 需要说明的是，这份结论有个前提：这里的矩阵只有 1024×1024，任务小到几乎全是"启动开销"，
# 所以打包才这么划算。矩阵更大、序列更长、模型层数更多时，计算本身占主导，批处理的收益会小一些，
# 但绝不会消失——因为批处理始终是在用同一批硬件资源同时干更多的活。