"""
本当に最小のAI

仕組み: 「直前の8文字」→ Attentionで文脈を集める → FFNで加工 →「次の1文字」を予測
（文脈参照ver: Self-Attention + FFN）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ハイパーパラメータ ---
block_size = 8   # 一度に見る文字数（最小verは1）
n_embd = 8       # 各文字を表すベクトルの次元数

# --- 学習データ ---
text = "ももたろうはももからうまれた"

# --- 文字と数値の対応表 ---
chars = sorted(set(text))
vocab = len(chars)  # 文字の種類数
to_i = {c: i for i, c in enumerate(chars)}
to_c = {i: c for i, c in enumerate(chars)}

print(f"文字数: {len(text)}  語彙: {vocab}種 {chars}")

# --- 学習データを作る ---
# 最小ver: (1文字, 次の1文字) のペア
# 位置認識ver: (8文字, それぞれの次の1文字) のペア
#
# 例: 入力 "ももたろうはもも" → 正解 "もたろうはももか"
#     位置1の"も"→正解"も", 位置2の"も"→正解"た", ... 位置8の"も"→正解"か"
X_list, Y_list = [], []
for i in range(len(text) - block_size):
    X_list.append([to_i[c] for c in text[i:i+block_size]])
    Y_list.append([to_i[c] for c in text[i+1:i+block_size+1]])

X = torch.tensor(X_list)  # (サンプル数, 8)
Y = torch.tensor(Y_list)  # (サンプル数, 8)

print(f"学習サンプル数: {X.shape[0]}  入力の形: {list(X.shape)}")

# --- モデル ---
tok_emb = nn.Embedding(vocab, n_embd)       # 各文字を8次元のベクトルに変換
pos_emb = nn.Embedding(block_size, n_embd)  # 各位置を8次元のベクトルに変換

# Self-Attention（文脈参照verで追加）
# 各位置が「過去のどの位置に注目するか」を学んで、情報を集める
query = nn.Linear(n_embd, n_embd, bias=False)  # 「何を探しているか」
key   = nn.Linear(n_embd, n_embd, bias=False)  # 「何を持っているか」
value = nn.Linear(n_embd, n_embd, bias=False)  # 「実際に渡す情報」
# causal mask: 未来の文字を見えなくする三角行列
tril = torch.tril(torch.ones(block_size, block_size))

# Feed-Forward Network
# Attentionが集めた情報を非線形に加工する
ffn = nn.Sequential(
    nn.Linear(n_embd, n_embd * 4),  # 8次元 → 32次元に拡大
    nn.ReLU(),                       # 負の値を0にする（これが非線形を生む）
    nn.Linear(n_embd * 4, n_embd),  # 32次元 → 8次元に戻す
)

net = nn.Linear(n_embd, vocab)              # そのベクトルから次の文字を予測
loss_fn = nn.CrossEntropyLoss()
all_params = (list(tok_emb.parameters()) + list(pos_emb.parameters())
              + list(query.parameters()) + list(key.parameters()) + list(value.parameters())
              + list(ffn.parameters()) + list(net.parameters()))
opt = torch.optim.Adam(all_params, lr=0.01)  # SGDより安定して学習が進むオプティマイザ

params = sum(p.numel() for p in all_params)
print(f"パラメータ数: {params}\n")


def self_attention(x):
    """各位置が過去の位置の情報を集める"""
    T = x.shape[-2]  # 文字数（学習時は8、生成時は1〜8）
    q = query(x)     # 各位置の「何を探しているか」
    k = key(x)       # 各位置の「何を持っているか」
    v = value(x)     # 各位置の「実際に渡す情報」

    # QueryとKeyの内積で「どの位置にどれだけ注目するか」を計算
    score = q @ k.transpose(-2, -1) * (n_embd ** -0.5)  # スケーリング

    # 未来の位置を-infにして、softmaxで0にする（カンニング防止）
    score = score.masked_fill(tril[:T, :T] == 0, float("-inf"))
    weight = F.softmax(score, dim=-1)

    # 注目度に応じてValueの重み付き平均を取る → 文脈を反映したベクトル
    return weight @ v


# --- 学習 ---
positions = torch.arange(block_size)  # rangeのPyTorch版 → tensor([0, 1, 2, 3, 4, 5, 6, 7])

for step in range(500):
    t = tok_emb(X)                # 文字→ベクトル  (サンプル数, 8, 8)
    p = pos_emb(positions)        # 位置→ベクトル  (8, 8)
    x = t + p                     # 合体           (サンプル数, 8, 8)
    x = self_attention(x)         # 過去の文脈を集める（文脈参照verで追加）
    x = ffn(x)                    # 集めた情報を非線形に加工
    pred = net(x)                 # 予測           (サンプル数, 8, vocab)
    B, T, V = pred.shape
    loss = loss_fn(pred.view(B*T, V), Y.view(B*T))  # 全位置まとめて誤差計算
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"  step {step:3d}  loss={loss.item():.3f}")

# --- 生成: 文脈を伸ばしながら1文字ずつ予測 ---
print("\n--- 生成 ---")
idx = [to_i["も"]]  # 最初の1文字から始める

with torch.no_grad():
    for _ in range(15):
        # 直近block_size文字を入力にする（足りなければある分だけ）
        context = idx[-block_size:]
        x = torch.tensor([context])            # (1, 文字数)
        t = tok_emb(x)                         # (1, 文字数, 8)
        p = pos_emb(torch.arange(len(context)))  # (文字数, 8)
        h = t + p
        h = self_attention(h)                  # 過去の文脈を集める
        h = ffn(h)                             # 集めた情報を非線形に加工
        logits = net(h)                        # (1, 文字数, vocab)
        logits = logits[0, -1, :]              # 最後の位置の予測だけ使う
        probs = F.softmax(logits, dim=-1)
        next_i = torch.multinomial(probs, 1).item()
        idx.append(next_i)

result = "".join([to_c[i] for i in idx])
print(result)
