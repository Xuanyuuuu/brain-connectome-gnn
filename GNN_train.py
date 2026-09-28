import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
import copy
from datetime import datetime
import time

from config import data_store
from GNN_data import target, train_graphs, val_graphs, test_graphs, top_fraction,network_id,network_name,compo_id,compo_name
from GNN_model import BrainGNN



target = str(target.name)  # taken from GNN_data, so it can't drift
hidden = 16
dropout = 0.4
lr = 1e-3
weight_decay = 1      # was 1e-2: too weak for AdamW (|w| ~ 1/wd before decay bites); now regularises graph branch + head
global_weight_decay = 1  # separate, stronger L2 on global_mlp (LR needed C=0.01); try 1e-1, then 1
batch_size = 32
max_epochs = 200
patience = 30             # stop after this many epochs without a val AUC gain
seed = 44              # add more (e.g. 43, 44) once one run looks sane
readout = "compo"
conv_hidden = 4
graph = "full" if top_fraction == 1.0 else f"top{round(top_fraction * 100)}"
graph += f"_{readout}"   

use_graph = True
use_global = True

if use_graph and use_global:
    branches = "both"
elif use_graph:
    branches = "graph_only"
else:
    branches = "global_only"

LR_BASELINE_VAL_AUC = 0.675       # logistic regression, gender, C=0.01

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def choose_threshold(y,p):
    thresholds = np.linspace(0.05,0.99,95)
    f1s = []
    for t in thresholds:
        pred = p>=t
        f1_temp = f1_score(y,pred,zero_division=0)
        f1s.append(f1_temp)

    index = np.argmax(f1s)
    return thresholds[index]

set_seed(seed)

train_loader = DataLoader(train_graphs,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_graphs,batch_size=batch_size)
test_loader = DataLoader(test_graphs,batch_size=batch_size)

model = BrainGNN(
    node_dim=train_graphs[0].x.shape[1],
    global_dim=train_graphs[0].global_x.shape[1],
    hidden=hidden,
    dropout=dropout,
    use_graph=use_graph,
    use_global=use_global,
    conv_hidden=conv_hidden,
    readout=readout
)

criterion = nn.BCEWithLogitsLoss()

# Two parameter groups so the global branch can be regularised more strongly.
# Names come from the attributes in BrainGNN.__init__ (e.g. "global_mlp.weight", "conv1.lin.weight").
global_params = [p for name, p in model.named_parameters() if name.startswith("global_")]
other_params = [p for name, p in model.named_parameters() if not name.startswith("global_")]
optimizer = torch.optim.AdamW([
    {"params": other_params, "weight_decay": weight_decay},
    {"params": global_params, "weight_decay": global_weight_decay},
], lr=lr)



history = []
best_auc, best_epoch, best_state, best_val_prob = -1.0, 0, None, None

start = time.perf_counter()
run_dir = data_store/'GNN_Run'/f'{graph}_{branches}_gwd{global_weight_decay:g}_{datetime.now():%m%d_%H%M%S}'
run_dir.mkdir(parents=True)

for epoch in range(1,max_epochs+1):
    epoch_start = time.perf_counter()
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits,batch.y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()*batch.num_graphs

    model.eval()
    total_val_loss = 0
    val_probs = []
    y_val = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_logits = model(val_batch)
            val_loss = criterion(val_logits,val_batch.y)
            total_val_loss += val_loss.item() * val_batch.num_graphs
            probs = torch.sigmoid(val_logits)
            y_val.append(val_batch.y)
            val_probs.append(probs)
    y_val = torch.cat(y_val).numpy()
    p_val = torch.cat(val_probs).numpy()
    val_auc = roc_auc_score(y_val,p_val)

    improved = val_auc > best_auc
    if improved:
        best_auc = val_auc
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        best_val_prob = p_val

    history.append({
        'seeds':seed,
        'epoch':epoch,
        'train_loss':total_train_loss/len(train_graphs),
        'val_loss':total_val_loss/len(val_graphs),
        'val_auc':val_auc,
    })
    epoch_seconds = time.perf_counter() - epoch_start
    print(
        f"Epoch{epoch}/{max_epochs} | "
        f"train_loss{total_train_loss/len(train_graphs):.4f} | "
        f"val_loss: {total_val_loss/len(val_graphs):.4f} | "
        f"val_auc {val_auc:.4f}{'*' if improved else " "} | "
        f"best_auc{best_auc:.4f} @ {best_epoch} epoch | "
        f"{epoch_seconds:.1f}s/epoch | elapsed {(time.perf_counter() - start)/60:.1f} min",
        flush=True
    )

    if epoch - best_epoch >= patience:
        print(f"Early stop: no val AUC gain after {patience} epochs")
        break



model.load_state_dict(best_state)
torch.save(best_state,run_dir/f"model_seed{seed}.pt")
threshold = choose_threshold(y_val,best_val_prob)


model.eval()

total_test_loss = 0
test_probs = []
y_test = []
with torch.no_grad():
    for test_batch in test_loader:
        test_logits = model(test_batch)
        test_loss = criterion(test_logits,test_batch.y)
        total_test_loss += test_loss.item() * test_batch.num_graphs
        probs = torch.sigmoid(test_logits)
        test_probs.append(probs)
        y_test.append(test_batch.y)

y_test = torch.cat(y_test).numpy()
p_test_prob = torch.cat(test_probs).numpy()
test_auc = roc_auc_score(y_test,p_test_prob)
average_precision = average_precision_score(y_test,p_test_prob)

p_test = (p_test_prob >= threshold).astype(int)
accuracy = accuracy_score(y_test,p_test)
f1 = f1_score(y_test,p_test,zero_division=0)



print(
    f"Test AUC: {test_auc:.4f} | "
    f"Average Precision: {average_precision:.4f} | "
    f"Accuracy: {accuracy:.4f} | "
    f"F1: {f1:.4f} | "
    f"Threshold: {threshold:.4f}",
    flush=True,
)

history = pd.DataFrame(history)
fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 4))
ax_loss.plot(history["epoch"], history["train_loss"], label="train")
ax_loss.plot(history["epoch"], history["val_loss"], label="val")
ax_loss.set(xlabel="epoch", ylabel="BCE loss", title="Loss")
ax_loss.legend()
ax_auc.plot(history["epoch"], history["val_auc"], label="GNN val")
ax_auc.axhline(LR_BASELINE_VAL_AUC, linestyle="--", color="gray", label="LR baseline")
ax_auc.set(xlabel="epoch", ylabel="ROC-AUC", title="Validation AUC")
ax_auc.legend()
fig.tight_layout()
fig.savefig(run_dir/'curves.png', dpi=150)
plt.close(fig)


history.to_csv(run_dir / "history.csv", index=False)

val_pred = (best_val_prob >= threshold).astype(int)
pd.DataFrame([
    {"split": "val", "roc_auc": best_auc,
     "ap": average_precision_score(y_val, best_val_prob),
     "accuracy": accuracy_score(y_val, val_pred),
     "f1": f1_score(y_val, val_pred, zero_division=0)},
    {"split": "test", "roc_auc": test_auc, "ap": average_precision,
     "accuracy": accuracy, "f1": f1},
]).assign(seed=seed, best_epoch=best_epoch, threshold=threshold, graph=graph, branches=branches,
          hidden=hidden, readout=readout, conv_hidden=conv_hidden,dropout=dropout, lr=lr, weight_decay=weight_decay,
          global_weight_decay=global_weight_decay
).to_csv(run_dir / "metrics.csv", index=False)