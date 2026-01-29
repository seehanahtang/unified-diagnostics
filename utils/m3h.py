import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy

class HeadMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        # At least one layer is required. We output a vector of size hidden_dim.
        layers = []
        last = in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(last, hidden_dim), nn.ReLU(inplace=True)]
            last = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 

class M3H(nn.Module):
    def __init__(self, input_dim, y1_bins=6, alpha=0.1, hidden_dim=1, hidden_layers=1, feature_indices_per_head=None, prune_amount=0.5):
        super().__init__()
        self.y1_bins = y1_bins
        self.total_tasks = y1_bins  # only ordinal bins
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.alpha = alpha

        # Feature selection for each task head
        if feature_indices_per_head is None:
            self.feature_indices_per_head = [list(range(input_dim)) for _ in range(self.total_tasks)]
        else:
            self.feature_indices_per_head = feature_indices_per_head

        # Task-specific projections
        self.task_heads = nn.ModuleList([
            HeadMLP(len(self.feature_indices_per_head[i]), self.hidden_dim, self.hidden_layers)
            for i in range(self.total_tasks)
        ])

        # Optional pruning on the first linear layer inside each head (common case)
        for head in self.task_heads:
            # If you want to prune all layers, loop through modules and check for nn.Linear
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    prune.l1_unstructured(m, name='weight', amount=prune_amount)
                    break  # prune only the first layer per head; remove this break to prune all

        # Attention mechanism
        self.task_queries = nn.Parameter(torch.randn(self.total_tasks, self.hidden_dim))
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Input-dependent bias (flexible)
        self.bias = nn.Parameter(torch.tensor(2.0))
    
    def thresholds(self):
        deltas = F.softplus(self._theta_deltas) + self.min_delta
        thetas = self._theta_shift + torch.cumsum(deltas, dim=0)
        return thetas    

    def forward(self, x):
        B = x.shape[0]

        # Step 1: Get task-specific features and pass through heads
        task_outputs = torch.stack([
            self.task_heads[i](x[:, self.feature_indices_per_head[i]])
            for i in range(self.total_tasks)
        ], dim=1)

        # Step 2: Self-attention across tasks
        K = self.key_proj(task_outputs)
        V = self.value_proj(task_outputs)
        Q = self.task_queries.unsqueeze(0).expand(B, -1, -1)

        attn_logits = torch.matmul(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        M_s_max = attn_logits.max(dim=2, keepdim=True).values + 1e-8
        M_s_normalized = attn_logits / M_s_max

        I_s = torch.eye(self.total_tasks, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        W_s = F.softmax(I_s + self.alpha * M_s_normalized, dim=2)

        self.last_attention = W_s.detach().mean(dim=0).cpu()  # For inspection

        attended = torch.matmul(W_s, V)  # shape: [B, T, H]

        # Step 3: Cumulative ordinal output
        y1_logits = attended.mean(dim=-1)  # shape: [B, T]
        positive = F.softplus(y1_logits)   # ensure monotonicity

        # Flexible bias per sample
        bias = self.bias
        y1_probs = torch.sigmoid(y1_logits - self.bias) 
        
        # Clamp for numerical stability
        y1_probs = torch.clamp(y1_probs, 1e-7, 1 - 1e-7)

        return y1_probs

# Loss for ordinal task only
def ordinal_loss(y1_pred, y1_true, weighted=False): 
    return F.binary_cross_entropy(y1_pred, y1_true, reduction='sum') / y1_pred.shape[0]

def multitask_bce_loss(
    preds: torch.Tensor,             # [B, T] logits
    targets: torch.Tensor,           # [B, T] in {0,1}, NaN -> missing/ignore
    *,
    from_logits: bool = True,        # keep True for training
) -> torch.Tensor:
    """
    1) elementwise BCE
    2) mean over samples per task (ignoring NaNs)
    3) equal-mean over tasks (or weighted by task_weights)

    If `task_weights` is provided, tasks with w>0 are averaged; others ignored.
    """
    B, T = preds.shape
    assert targets.shape == (B, T), "preds and targets must both be [B, T]"

    if from_logits:
        base = F.binary_cross_entropy_with_logits(preds, targets, reduction='none', pos_weight=None)
        elem = base
    else:
        probs = preds.clamp_min(1e-7).clamp_max(1 - 1e-7)
        elem = F.binary_cross_entropy(probs, y, reduction='none')

    # zero-out missing labels
    elem = elem * mask

    # per-task mean over available samples
    denom = mask.sum(dim=0).clamp_min(1)       # [T]
    task_means = elem.sum(dim=0) / denom       # [T]

    return task_means.mean()

def finalize_pruning(model):
    for head in model.task_heads:
        for m in head.modules():
            if isinstance(m, nn.Linear) and hasattr(m, 'weight_orig'):
                prune.remove(m, 'weight')

# Training loop
def train_with_early_stopping(model, train_loader, val_loader, optimizer, num_epochs=100, patience=5, l1=1e-4):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y1_batch in train_loader:
            optimizer.zero_grad()
            y1_pred = model(X_batch)
            # loss = ordinal_loss(y1_pred, y1_batch)
            loss = multitask_bce_loss(y1_pred, y1_batch)

            l1_reg = sum(
                param.abs().sum()
                for name, param in model.named_parameters()
                if "task_heads" in name
            )
            loss += l1 * l1_reg

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y1_batch in val_loader:
                y1_pred = model(X_batch)
                # loss = ordinal_loss(y1_pred, y1_batch)
                loss = multitask_bce_loss(y1_pred, y1_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    finalize_pruning(model)

def load_model_from_file(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model


# model = M3H(
#     input_dim=train_X.shape[1], 
#     y1_bins=y1_bins,
#     alpha=1,
#     prune_amount=0.5
# )

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train_with_early_stopping(model, train_loader, val_loader, optimizer, num_epochs=10000, patience=200, l1=1e-2)