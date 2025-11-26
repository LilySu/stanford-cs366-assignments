import torch
from transformers import SGD

def run_experiment(lr):
    print(f"\nTesting Learning Rate: {lr}")
    torch.manual_seed(42) # Ensure same initialization for all runs
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    
    losses = []
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
    print(f"Start Loss: {losses[0]:.4f} -> End Loss: {losses[-1]:.4f}")
    if losses[-1] > losses[0] * 10:
        print("Result: Diverged")
    elif losses[-1] < losses[0]:
        print("Result: Decayed")
    else:
        print("Result: Unstable/Oscillated")

if __name__ == "__main__":
    for lr in [1.0, 10.0, 100.0, 1000.0]:
        run_experiment(lr)