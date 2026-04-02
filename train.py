import os
import torch

from prepare_data import prepare_chikusei
from models.specresnet import Autoencoder3D
from losses.loss import total_loss as loss_function
from losses.loss import tv_loss_3d, sam_loss, charbonnier_loss


# ====== Process Single Batch ======= #

def process_batch(model, batch, device, lambda_tv, lambda_sam,
                  use_charbonnier=True, return_latents=False):

    data = batch.to(device)
    output = model(data)

    if isinstance(output, tuple):
        reconstructed, x1_quant = output
    else:
        reconstructed = output
        x1_quant = None

    loss = loss_function(
        reconstructed, data,
        lambda_tv=lambda_tv,
        lambda_sam=lambda_sam,
    )

    if return_latents:
        return loss, reconstructed, x1_quant
    else:
        return loss


# ====== One Training Epoch ====== #

def train_one_epoch(model, dataloader, optimizer, device,
                    lambda_tv=0.0, lambda_sam=0.0, use_charbonnier=True):

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        loss = process_batch(model, batch, device,
                             lambda_tv, lambda_sam, use_charbonnier)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ====== Validation Epoch ======= #

def validate(model, dataloader, device,
             lambda_tv=0.0, lambda_sam=0.0, use_charbonnier=True):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            loss = process_batch(model, batch, device,
                                 lambda_tv, lambda_sam, use_charbonnier)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ======== Full Training Loop ========= #

def train_model(model, train_loader, val_loader, optimizer,
                num_epochs=300, patience=10,
                lambda_tv=0.0, lambda_sam=0.0, use_charbonnier=True,
                device='cuda', save_path=None, scheduler=None):

    print(f"Training with lambda_tv={lambda_tv}, lambda_sam={lambda_sam}")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model.to(device)

    for epoch in range(num_epochs):

        if hasattr(model, 'quantizer') and hasattr(model.quantizer, 'set_epoch'):
            model.quantizer.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     lambda_tv, lambda_sam, use_charbonnier)

        val_loss = validate(model, val_loader, device,
                            lambda_tv, lambda_sam, use_charbonnier)

        print(f"Epoch {epoch+1:03}/{num_epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
