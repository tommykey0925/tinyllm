"""
本当に最小のAI - たった1層のニューラルネットワーク

仕組み: 「直前の8文字」→「次の1文字」を予測する
（最小verは1文字→1文字だった）
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
# 文脈付与ver: (8文字, それぞれの次の1文字) のペア
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

# --- モデル: Token Embedding + Positional Embedding → Linear ---
tok_emb = nn.Embedding(vocab, n_embd)       # 各文字を8次元のベクトルに変換
pos_emb = nn.Embedding(block_size, n_embd)  # 各位置を8次元のベクトルに変換（文脈付与verで追加）
net = nn.Linear(n_embd, vocab)              # そのベクトルから次の文字を予測
loss_fn = nn.CrossEntropyLoss()
all_params = list(tok_emb.parameters()) + list(pos_emb.parameters()) + list(net.parameters())
opt = torch.optim.SGD(all_params, lr=1.0)

params = sum(p.numel() for p in all_params)
print(f"パラメータ数: {params}\n")

# --- 学習 ---
positions = torch.arange(block_size)  # rangeのPyTorch版 → tensor([0, 1, 2, 3, 4, 5, 6, 7])

for step in range(300):
    t = tok_emb(X)                # 文字→ベクトル  (サンプル数, 8, 8)
    p = pos_emb(positions)        # 位置→ベクトル  (8, 8)
    x = t + p                     # 合体           (サンプル数, 8, 8)
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
        logits = net(h)                        # (1, 文字数, vocab)
        logits = logits[0, -1, :]              # 最後の位置の予測だけ使う
        probs = F.softmax(logits, dim=-1)
        next_i = torch.multinomial(probs, 1).item()
        idx.append(next_i)

result = "".join([to_c[i] for i in idx])
print(result)
