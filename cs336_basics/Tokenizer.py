import json  # 读 vocab.json 用；讲义要求从训练产出的两个文件里还原分词器
import regex as re  # 第三方正则库，比 Python 自带的 re 多认得 \p{L} 这种"Unicode 字符类别"，GPT-2 的切词规则没它玩不转
from collections.abc import Iterable  # 借来做类型标注的，告诉读代码的人：这个方法不挑食，任何能 for 循环的东西都能传


# 下面这段模块级说明是留给维护者的"设计便签"：先讲清楚特殊 Token 的整体处理思路，
# 后来人改 encode 之前看它一眼，就知道为什么要单独搞一套特殊标记的匹配逻辑
"""
For special_tokens:
    推理/编码阶段 (Tokenizer.encode)
        在模型使用分词器将文本转为 ID 时，必须优先匹配特殊 Token。
    代码逻辑：
        正则匹配：构建一个包含所有特殊 Token 的正则表达式。
        优先级：先扫描文本，一旦发现特殊 Token，直接将其转为对应的 ID。
        普通处理：特殊 Token 之间的文本，再走正常的 GPT-2 预分词和 BPE 合并流程。
"""

class BPETokenizer:  # 分词器类，大驼峰命名是 Python 的惯例；以后"文本变 ID、ID 变文本"的活儿全包在它身上
    # 类的说明书（docstring）：鼠标悬停或按 help() 时能看到，先说清这个类是什么、能干什么
    """
    字节级 BPE（Byte-Pair Encoding）分词器实现。
    
    该分词器将任意字符串编码为整数 ID 序列，并能将 ID 序列还原。
    它采用字节级处理，确保不会出现未知词（OOV）错误。
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):  # 构造器：每 new 一个分词器实例，都得把"词表、合并规则、特殊标记"喂齐，后两样甚至可以不给
        # 构造函数的参数说明书：逐个解释三个入参的格式和含义，后来维护的人照着他准备数据就行
        """
        初始化分词器。
        
        参数:
            vocab: 词汇表，建立整数 ID 到 字节块(bytes) 的映射。
            merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b)，
                   表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序。
            special_tokens: 特殊标记列表（如 <|endoftext|>），这些标记不会被 BPE 规则拆分。
        """
        # 1. 建立双向映射，方便查表
        self.vocab = vocab  # ID -> 字节块；外面训练好的词表原样存下，解码时主要查它
        self.id_to_byte = vocab  # 同一张表再起个名字，图个可读性——光看变量名就知道是"拿 ID 换字节"用的；两个名字指向同一个对象，不占额外内存
        self.byte_to_id = {v: k for k, v in vocab.items()} # 字节块 -> ID；关键一步：把词表整个翻转过来，因为编码时手里握的是字节块，得反查 ID；字典推导式一行翻转，比写 for 循环清爽
        
        # 2. 将合并规则转换为Rank字典。
        # BPE 编码时，必须优先应用在训练阶段较早出现的合并规则。
        # 字典结构为: {(byte_a, byte_b): 顺序索引}
        self.merges = {pair: i for i, pair in enumerate(merges)}  # 训练给的 merges 是有先后的列表，这里给它编号成"规则 -> 名次"；名次越小说明越早学到（频率越高），编码时就越得先用它；enumerate 顺手把下标送出来当名次
        
        self.special_tokens = special_tokens or []  # 没传特殊标记时给个空列表兜底，后面就不用到处判 None；or 的妙用是 None 和 [] 统统变成 []
        
        # 3. 构建特殊 Token 的正则表达式
        if self.special_tokens:  # 有特殊标记才值得费这个劲；一个都没有的话，encode 会走快速通道，正则建了也白建
            # 关键：必须按照长度从长到短排序（reverse=True）。
            # 这样正则引擎会优先匹配最长的特殊标记，防止重叠标记（如 <|a|><|b|>）被错误拆分。
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)  # 按长度排队，长的站前排；不然 <|end|> 这种短标记可能抢先匹配，把 <|endoftext|> 拦腰咬断，控制指令就废了
            # 使用 re.escape 确保标记中的特殊字符（如 | 或 [ ）被当作普通字符处理
            special_pattern = "|".join(re.escape(t) for t in sorted_special)  # 用 | 把所有标记拼成"匹配这个或匹配那个"；re.escape 给标记里的 |、[ 这些正则通配符穿上 escape 盔甲，否则 <|endoftext|> 里的 | 会被正则当成"或"来解析，匹配直接错乱
            self.special_regex = re.compile(special_pattern)  # 编译成正则对象存起来；编译一次到处用，比每次编码都现编快得多
        else:
            self.special_regex = None  # 没有特殊标记就置空，encode 看到 None 就知道整套"剪文本找标记"的逻辑可以整个跳过

        # 4. GPT-2 官方预分词正则表达式。
        # 它的作用是在应用 BPE 合并前，先将文本切分成单词、标点、数字等逻辑块。
        # 这样做是为了防止 BPE 规则跨越单词或标点（例如：防止将 "dog" 的末尾和 "." 合并）。
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # 照抄 GPT-2 原版的切词正则：'s/'ll/'ve 这类缩略词单独一块，字母、数字、标点各自成块，空格还分"词前一个"和"连续多个"两种切法；先把块切好，BPE 合并就被圈在块里，绝不会把 "dog" 的尾巴和 "." 粘在一起

    @staticmethod
    def _bytes_to_unicode() -> dict[int, str]:  # GPT-2 的"字节 → 可见字符"对照表，序列化词表时要靠它把 bytes 变成能写进文本文件的字符
        """
        把 0~255 每个字节映射成一个看得见、能存进 json/txt 的字符。

        为什么要这么绕：BPE 是在字节层面做的，但词表要存成文本文件，
        而 0x00~0x20 这些控制字符没法直接写进去。
        GPT-2 的做法是把可打印的 256 个字节区间原样保留，
        剩下的 68 个不可见字节映射到 U+0100 之后的一串字符上
        （Ā = 0x00，ā = 0x01，……）。
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))   # 英文标点 + 数字 + 大小写字母
            + list(range(ord("¡"), ord("¬") + 1))  # 拉丁补充区
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]  # 这些字节直接映射到自己
        n = 0
        for b in range(256):
            if b not in bs:  # 剩下的不可见字节，依次分配 256+n 的码位
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, (chr(c) for c in cs)))

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "BPETokenizer":
        # 讲义要求的类方法：从 BPE 训练产出的 vocab.json + merges.txt 直接还原分词器。
        # 没有它的话，每次加载都得在外面手写一遍"反查字节表"，容易和训练端的保存逻辑走偏。
        """
        从序列化文件构造分词器。

        参数:
            vocab_filepath: vocab.json 的路径。格式是 {"0": "Ā", "1": "ā", ..., "9999": "<|endoftext|>"},
                            即 **id(字符串) -> byte-level 编码后的 token 字符串**。
                            key 是字符串化的整数，value 要经反查表才能变回 bytes。
            merges_filepath: merges.txt 的路径。每行两列，如 "Ġ t"，行号就是合并优先级。
                            首行如果是 "#version: ..." 之类的头部会被跳过。
            special_tokens: 特殊标记列表，默认 None。传了就会在编码时整体保留、不拆成子词。

        返回:
            可直接 encode/decode 的 BPETokenizer 实例。

        异常:
            KeyError：vocab 里出现了对照表之外的字符，通常意味着传进来的不是
                      GPT-2 格式（比如已经是 {id: bytes} 的 pickle）。
        """
        byte_encoder = cls._bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        def to_bytes(s: str) -> bytes:  # 每个字符查回一个字节
            return bytes(byte_decoder[c] for c in s)

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        # json 的 key 一定是字符串，这里转回 int；value 是 byte-level 字符串，要解开
        vocab = {int(k): to_bytes(v) for k, v in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) != 2:  # 跳过空行和版本头
                    continue
                merges.append((to_bytes(parts[0]), to_bytes(parts[1])))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        
    def encode(self, text: str) -> list[int]:  # 对外主入口：传字符串进去，吐一串整数 ID 出来
        # 方法说明书：把 encode 的内部职责拆成三点讲清楚，先读这段再看代码能少踩坑
        """
        将输入的原始字符串编码为整数 ID 列表。
        
        该方法的核心逻辑是：
        1. 作为一个“协调者”，它负责处理文本中的“特殊标记（Special Tokens）”和“普通文本”。
        2. 特殊标记（如 <|endoftext|>）被视为原子，直接映射为 ID，不参与 BPE 的拆分和合并。
        3. 普通文本片段则被交给底层逻辑执行预分词和 BPE 算法。
        
        参数:
            text: 需要编码的原始字符串（例如 "Hello<|end|>World"）。
            
        返回:
            list[int]: 编码后的整数 ID 序列。
        """
        # --- 步骤 1: 边界情况检查 ---
        # 如果输入是空字符串或 None，直接返回空列表。
        # 这是为了防止后续逻辑在处理空文本时产生错误。
        if not text:  # 空串和 None 在 Python 里都是"假值"，一个判断全拦下；不拦也能跑，但后面正则和切片纯属白忙一场
            return []  # 输入啥都没有，就还一个空列表，调用方拿到空结果也不会崩

        # --- 步骤 2: 情况 A - 快速路径 (Fast Path) ---
        # 如果我们在初始化时没有定义任何特殊标记（或者特殊标记列表为空），
        # 那么整个文本都可以被视为一段连续的“普通文本”。
        # 我们直接调用内部方法 _encode_text_segment 进行 BPE 处理并返回结果。
        if not self.special_regex:  # 初始化时没建出正则，说明压根没有特殊标记，走捷径最省事
            return self._encode_text_segment(text)  # 全文当成一大段普通文本，直接丢给核心 BPE 逻辑，省掉下面剪来剪去的全部开销

        # --- 步骤 3: 情况 B - 处理含有特殊标记的复杂文本 ---
        # 此时文本中可能混有普通文字和特殊标记，我们需要像“剪刀”一样把它们切开。
        tokens = []  # 准备一只空篮子，最终所有 ID 都往里扔；用局部变量，不碰实例状态，多次调用互不干扰
        
        # last_pos 用于记录上一次匹配结束的位置，帮助我们定位“特殊标记”之间的“缝隙”。
        last_pos = 0  # 一把"剪刀"的游标，记住上次剪到哪儿，才知道下一段普通文字从哪开头
        
        # 使用 finditer 遍历文本中所有符合特殊标记模式的匹配项。
        # finditer 的好处是它提供了 match.start() 和 match.end()，
        # 这让我们能够精确地知道特殊标记在哪里开始，在哪里结束。
        for match in self.special_regex.finditer(text):  # finditer 把文本里每个特殊标记连同位置一起交出来（不像 findall 只给字符串）；位置信息正是切出"缝隙"的关键
            
            # 3.1 提取并处理“前置普通文本”
            # 这里的区间是 [last_pos, match.start())。
            # " hello <|endoftext|> world"
            # 这段文本是夹在两个特殊标记之间（或者开头到第一个特殊标记之间）的普通文字。
            pre_text = text[last_pos:match.start()]  # 从上次剪到的地方剪到本次标记起点，中间夹的就是普通文字；两个标记挨着时这里是空串，下面会判掉
            
            # 如果这两个标记之间确实有文字（长度 > 0）
            if pre_text:  # 空串直接跳过，免得调一次函数处理空气
                # 调用核心 BPE 逻辑。_encode_text_segment 会执行：
                # 1. GPT-2 预分词正则切分。
                # 2. 字节化。
                # 3. 按照 merges 规则进行贪婪合并。
                tokens.extend(self._encode_text_segment(pre_text))  # 这段普通文字走正经 BPE 流程；用 extend 不用 append，是因为它返回的是一串 ID，得拆开平铺进篮子（append 会把整个列表当一个元素塞进去，结果就嵌套乱了）
                # pre_tokens : [1,2,3,...] self._encode_text_segment: [4,5,6] tokens.extend -> [1,2,3,...,4,5,6]
                # token.append() : [1,2,3,...,[4,5,6]]
            
            # 3.2 处理“当前特殊标记”
            # match.group() 拿到的就是被识别出来的特殊标记字符串（如 "<|endoftext|>"）。
            special_tok = match.group()  # group() 不带参数就是"整个匹配到的内容"，这里即 <|endoftext|> 这个字符串本身
            
            # 核心原则：特殊标记不参与 BPE 合并！
            # 我们直接将其编码为 UTF-8 字节，然后在词表中查找其 ID。
            # 注意：这些标记在 train_bpe 阶段必须已经被手动加入到了词表中。
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])  # 特殊标记是原子，谁也不许拆：按 UTF-8 编成字节后直接查 ID；前提是训练阶段已经把它加进词表，否则这里会 KeyError 当场罢工——这正是开头便签里强调"必须手动加入词表"的原因
            
            # 3.3 更新游标
            # 将游标移动到当前匹配项的末尾，为寻找下一个片段做准备。
            last_pos = match.end()  # 剪刀往前挪到这个标记的末尾，下一轮要处理的就是它后面的文字
            
        # --- 步骤 4: 处理“收尾文本” ---
        # 如果最后一个特殊标记后面还有文字（例如 "Hello<|end|>World" 中的 "World"），
        # 或者整个文本根本没有特殊标记匹配（虽然逻辑上 Case A 已处理，但这里是双重保险），
        # 我们需要处理从 last_pos 到字符串末尾的所有剩余字符。
        remaining_text = text[last_pos:]  # 最后一个标记之后直到结尾的所有文字；切片不写结束位置就是"一直到头"，偷懒写法但很稳
        
        if remaining_text:  # 有尾巴才处理；文本以标记结尾时这里是空串，跳过
            # 剩余部分同样作为普通文本片段进行 BPE 编码。
            tokens.extend(self._encode_text_segment(remaining_text))  # 收尾文字和前面一视同仁，照样走 BPE；extend 平铺进结果，顺序不乱
            
        # 返回拼接好的所有 ID 列表
        return tokens  # 大功告成，整串 ID 交还给调用方；能走到这说明文本里有特殊标记，且已被完整切分处理

    def _encode_text_segment(self, text: str) -> list[int]:  # 类的"私活"方法：名字带下划线开头是 Python 里"内部人员专用"的约定，只处理不含特殊标记的纯文本片段
        # 一句话说明书：这个内部方法才是整个分词器真正干活的地方
        """
        内部核心函数：对不含特殊 Token 的纯文本片段应用 BPE 合并逻辑。
        """
        ids = []  # 这只小篮子专给当前文本片段攒 ID，和 encode 里那只各管各的，互不掺和
        
        # 使用 GPT-2 正则进行预分词，将文本拆成单词/标点符号块
        # 例如："Hello world!" -> ["Hello", " world", "!"]
        pre_tokens = self.gpt2_pat.findall(text)  # 拿 GPT-2 正则把片段切成一嘟噜小块，比如 ["Hello", " world", "!"]；先切块是为了守住"合并不跨词"的底线
        
        for p_tok in pre_tokens:  # 一个块一个块收拾，块与块之间老死不相往来，谁也不许把手伸进邻居家里
            # 第一步：将当前片段转为字节序列，并将每个字节看作一个独立的“部分（Part）”
            # 例如："Hello" -> [b'H', b'e', b'l', b'l', b'o']
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]  # 先把小块按 UTF-8 编成字节，再把每个字节包成单字节 bytes；byte-level 的底气就在这——任何字符最终都落成 0-255 的字节，不可能有字不在词表里
            
            # 第二步：反复执行合并，直到没有符合条件的合并规则为止
            while len(byte_parts) >= 2:  # 只剩一个部分时没得可合，循环到这打住；>=2 是合并的最低门槛
                # 在当前序列的所有相邻对中，寻找合并优先级最高（Rank 最小）的一对，即按照构造merge时添加pair的顺序进行合并
                best_pair = None  # 本轮"选美比赛"的冠军暂时空缺
                min_rank = float('inf')  # 用"无穷大"当初始门槛，任何真实名次都能轻松刷低它
                
                for i in range(len(byte_parts) - 1):  # 下标从 0 扫到倒数第二个（-1 是因为最后一个没有右邻居，配不成对）
                    pair = (byte_parts[i], byte_parts[i+1])  # 当前项和它的右邻居打包成一对，准备参赛
                    if pair in self.merges:  # 这对在训练规则里吗？不在的直接出局，连名次都不用查
                        rank = self.merges[pair]  # 查它的名次；数字越小越是早期学到的高频规则
                        if rank < min_rank:  # 比现任冠军还靠前？冠军换人
                            min_rank = rank  # 记下新门槛，后来者想上位得比它还小
                            best_pair = pair  # 新冠军上位，牢牢记住是哪一对
                
                # 如果找不到任何可以合并的规则，退出当前片段的合并过程
                if best_pair is None:  # 扫完一轮冠军还空缺，说明剩下的相邻对都没学过，这个块已经合并到头
                    break  # 跳出 while，别跟这个块死磕；break 只退当前块的合并，不影响下一个块
                
                # 第三步：执行合并操作。
                # 遍历当前序列，将所有出现的 best_pair 替换成合并后的长字节块。
                new_byte_parts = []  # 开一个空列表装合并后的新序列；不敢在原列表上动刀，怕边改边漏、下标错乱
                i = 0  # 遍历游标从最左边起步
                # [b'H', b'e', b'l', b'l', b'o', b'H', b'e'] -> [b'He', b'l', b'l', b'o', b'He']
                while i < len(byte_parts):  # 从左到右把整个序列过一遍
                    # 如果当前两个部分匹配最高优规则
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:  # 当前位置加右邻居正好是冠军组合，就合体；前面 i < len-1 是防越界——最后一个元素没右邻居可看，硬看会 IndexError
                        new_byte_parts.append(best_pair[0] + best_pair[1])  # 冠军组合前半截拼后半截，变成一个更长的字节块进新列表
                        i += 2 # 跳过下一项，因为已经合并了；右邻居已经被吃进合并块，不能再单独看它
                    else:
                        new_byte_parts.append(byte_parts[i])  # 没资格参与合并的，原样搬进新列表
                        i += 1  # 正常往前挪一格，继续看下一对
                byte_parts = new_byte_parts # 更新序列，进入下一轮 while 循环；新序列换上，接着找下一个能合的冠军
            
            # 第四步：将合并到极限后的所有字节块转换为词表中的 ID
            for part in byte_parts:  # 合并到头了，剩下的每个字节块都是词表里的正式居民
                ids.append(self.byte_to_id[part])  # 挨个查表换成 ID 攒起来；正常情况下都查得到，训练时每个中间产物都入过表
                
        return ids  # 这个片段的全部 ID，交回给 encode 去汇总

    def decode(self, ids: list[int]) -> str:  # 编码的逆操作：进一串 ID，出还原好的字符串
        # 方法说明书：一句话——把 ID 列表还原成原始文本
        """
        将 ID 列表解码为原始字符串。
        """
        # 1. 根据 ID 查表找回字节块
        byte_segments = [self.id_to_byte[i] for i in ids]  # 列表推导式逐个查表，把 ID 还原成它对应的字节块；顺序千万不能乱，乱了拼出来的字就面目全非
        
        # 2. 将所有字节块按顺序拼接成一个完整的字节流
        full_bytes = b"".join(byte_segments)  # 用空 bytes 当胶水，把所有块粘成一整条字节流；一个字符可能横跨好几个字节块，必须先拼回整体再解码
        
        # 3. 将字节流解码为 UTF-8 字符串。
        # 使用 errors="replace" 非常关键：因为 BPE 可能会生成不完整的字节序列
        # （例如 3 字节的中文字符只产生了一部分），此时不报错而是插入替换符（）。
        return full_bytes.decode("utf-8", errors="replace")  # 字节流按 UTF-8 翻译回人类文字；errors="replace" 是根保险丝——万一字节流被截断（比如中文 3 字节只剩 1 个），不炸异常，而是塞个 U+FFFD 替换字符占位，保证程序不崩、其余内容还能看

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:  # 为超大文件准备的迭代版编码器：来多少行吐多少 ID，绝不把整个文件吃进内存
        # 方法说明书：强调"内存友好"这个卖点，顺便交代参数和返回值的形态
        """
        内存高效的迭代编码器。
        
        参数:
            iterable: 一个可迭代的字符串对象（例如文件句柄）。
        返回:
            一个生成器，逐个产出编码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        # 【改】块尾的连续空白先扣下来，留给下一块再编。
        # 起因是 GPT-2 预分词正则里有一支 \s+(?!\S)：它专门把"后面不再跟非空白"的空白
        # 单独切出来。同一段空白出现在"块尾"和"块中间"时上下文不同，切法就可能不同。
        # 实测三连换行 "hello\n\n\nworld"：
        #   整篇编码 -> [31373, 628, 198, 6894]（628 是把两个换行并成的一个 token）
        #   逐行编码 -> [31373, 198, 198, 198, 6894]（三个独立换行）
        # decode 都对，但和"整篇编码"对不上，拿去做等值断言就会翻车。
        # 把块尾空白缓冲到下一块，两边的切法就一致了；代价只是多存一小段空白，
        # 不影响"内存友好"这个初衷。
        pending = ""  # 上一块没消化完的尾巴（只会是空白）
        for chunk in iterable:  # 可迭代对象喂什么就处理什么——文件句柄（一行一行来）、字符串列表都行，不挑食
            text = pending + chunk  # 先把上一块的尾巴接回来，让正则看到完整上下文
            trailing = re.search(r"\s+\Z", text)  # 找结尾那段连续空白；用 \Z 而不是 $，免得被"末尾换行"的特殊规则坑到
            if trailing:  # 结尾确实有空白：切下来存着，剩下的先编码
                pending = text[trailing.start():]
                text = text[:trailing.start()]
            else:  # 结尾不是空白，那就没有要留给下一块的东西
                pending = ""
            if text:  # 空串就别白调用一次了
                yield from self.encode(text)  # yield 是"生产一个吐一个"的节奏，不一次性攒齐所有结果，内存占用低；from 把 encode 返回的列表摊开了一个个吐，调用方拿到的就是一串连续不断的 ID
        # 最后一块的尾巴后面不会再有新内容，直接编掉，别丢字
        if pending:
            yield from self.encode(pending)
