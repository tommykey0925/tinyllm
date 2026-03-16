"""
本当に最小のAI - たった1層のニューラルネットワーク

仕組み: 「直前の1文字」→「次の1文字」を予測するだけ
"""

import torch
import torch.nn as nn

# --- 学習データ ---
text = "ももたろうはももからうまれた"

# --- 文字と数値の対応表 ---
chars = sorted(set(text))
vocab = len(chars)  # 文字の種類数
to_i = {c: i for i, c in enumerate(chars)}
to_c = {i: c for i, c in enumerate(chars)}

print(f"文字数: {len(text)}  語彙: {vocab}種 {chars}")

# --- 学習データを作る: (入力1文字, 正解=次の1文字) のペア ---
X = torch.tensor([to_i[c] for c in text[:-1]])  # 入力
Y = torch.tensor([to_i[c] for c in text[1:]])   # 正解

# --- モデル: Embedding → Linear の2層だけ ---
emb = nn.Embedding(vocab, 8)   # 各文字を8次元のベクトルに変換
net = nn.Linear(8, vocab)      # そのベクトルから次の文字を予測
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(list(emb.parameters()) + list(net.parameters()), lr=1.0)

params = sum(p.numel() for p in list(emb.parameters()) + list(net.parameters()))
print(f"パラメータ数: {params}\n")

# --- 学習 ---
for step in range(300):
    pred = net(emb(X))          # 予測
    loss = loss_fn(pred, Y)     # 正解との誤差
    opt.zero_grad()             # 勾配リセット
    loss.backward()             # 誤差逆伝播
    opt.step()                  # 重み更新
    if step % 100 == 0:
        print(f"  step {step:3d}  loss={loss.item():.3f}")

# --- 生成: 1文字ずつ予測を繰り返す ---
print("\n--- 生成 ---")
ch = "も"
result = ch
for _ in range(15):
    x = torch.tensor([to_i[ch]])
    logits = net(emb(x))
    probs = torch.softmax(logits, dim=-1)
    ch = to_c[torch.multinomial(probs, 1).item()]
    result += ch
print(result)
