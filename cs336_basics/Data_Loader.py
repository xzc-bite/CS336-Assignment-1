import numpy as np
import torch


class DataLoader:
    """语言模型的下一个 token 数据加载器。

    data 可以是一维的 list、np.ndarray 或 np.memmap：
    - list 会被转换成 np.ndarray；
    - np.memmap 不会被整体读入内存，适合大语料。
    """

    def __init__(self, data, batch_size: int, context_length: int, shuffle: bool = True):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.context_length = context_length
        self.shuffle = shuffle

        # 合法起点是 0 .. num_starts-1，一共 num_starts 个。
        self.num_starts = self.data_len - self.context_length
        self._cursor = 0

        if self.num_starts <= 0:
            raise ValueError("数据太短：至少需要 context_length + 1 个 token")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

    def _make_batch(self, starts, device):
        """starts: (B,)；返回 x, y，形状都是 (B, context_length)。"""
        idx = starts[:, None] + np.arange(self.context_length)
        x = torch.from_numpy(np.asarray(self.data[idx], dtype=np.int64))
        y = torch.from_numpy(np.asarray(self.data[idx + 1], dtype=np.int64))
        return x.to(device), y.to(device)

    def get_train_batch_data(self, device="cpu"):
        """随机（或顺序）获取一个 batch。"""
        if self.shuffle:
            # 高位不包含，因此结果正好覆盖 0 .. num_starts-1。
            starts = np.random.randint(0, self.num_starts, size=self.batch_size)
        else:
            # 顺序模式：从上次的位置继续，走到末尾后回到开头。
            starts = (self._cursor + np.arange(self.batch_size)) % self.num_starts
            self._cursor = (self._cursor + self.batch_size) % self.num_starts
        return self._make_batch(starts, device)

    def get_valid_batch_data_iter(self, device="cpu"):
        """按顺序遍历验证集：起点按 context_length 步进，窗口互不重叠。

        为什么不是每个位置都取一次（步长 1）：T=256、B=32 时相邻两批会有 31 个
        位置完全相同，同一段文本最多会被评估 256 次，val_loss 被重复样本主导。
        按 T 步进则每条样本只覆盖一段互不重叠的文本，看到的都是新内容。

        代价是批次数变少（约 data_len / (T*B)），一轮覆盖的 token 也少一些。
        最后一个 batch 可能不满 batch_size，下游按 vx.shape 算就行。
        """
        window_starts = np.arange(0, self.num_starts, self.context_length)
        for i in range(0, len(window_starts), self.batch_size):
            starts = window_starts[i:i + self.batch_size]
            yield self._make_batch(starts, device)

    def __len__(self):
        """返回验证集 batch 数（窗口互不重叠，最后一个 batch 可能不满）。"""
        windows = len(range(0, self.num_starts, self.context_length))
        return (windows + self.batch_size - 1) // self.batch_size
