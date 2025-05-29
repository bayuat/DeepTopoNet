from torch.utils.data import DataLoader
import h5py
import torch
from model import BedTopoCNN
from feature_augmentation import gradient_covariates, trend_surface
from dataset import BedTopoDataset
from loss import custom_loss
import numpy as np
import pandas as pd
import torch.optim as optim


import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    variables = ['surf_SMB', 'surf_dhdt', 'surf_elv', 'surf_vx', 'surf_vy', 'surf_x', 'surf_y']    
    data_dict = {}
    with h5py.File('./data/hackathon.h5', 'r') as f:
        for var in variables:
            data_dict[var] = np.flipud(np.transpose(f[var][0:600, 600:1200])).copy()

    # Stack the first 5 variables as input features
    input_variables = ['surf_SMB', 'surf_dhdt', 'surf_elv', 'surf_vx', 'surf_vy']
    inputs = np.stack([data_dict[var] for var in input_variables], axis=0)  # Shape: (5, 600, 600)
    gradient_features = gradient_covariates(inputs)  # Gradient shape: (num_features * 2, height, width)
    inputs = np.vstack([inputs, gradient_features])  # Concatenate gradients as additional features

    trend_features = []
    for i in range(inputs.shape[0]):  # Loop over all features
        trend = trend_surface(inputs[i])  # Compute trend surface for the feature
        trend_features.append(trend)

    trend_features = np.vstack(trend_features)  # Combine all trend features
    inputs = np.vstack([inputs, trend_features])  # Add trend features to inputs
    mean_inputs = inputs.mean(axis=(1, 2), keepdims=True)
    std_inputs = inputs.std(axis=(1, 2), keepdims=True)
    inputs = (inputs - mean_inputs) / std_inputs

    with h5py.File('./data/bed_BedMachine.h5', 'r') as file:
        bedmachine_data = np.flipud(np.transpose(file['bed_BedMachine'][0:600, 600:1200])).copy()
    target_bed = bedmachine_data
    mean_target = np.mean(target_bed)
    std_target = np.std(target_bed)
    target_bed = (target_bed - mean_target) / std_target

    surf_x_min = np.min(np.abs(data_dict['surf_x']))
    surf_y_min = np.min(np.abs(data_dict['surf_y']))    
    radar_mask = np.zeros(target_bed.shape, dtype=bool)    
    full_data_df = pd.read_csv('./data/data_full.csv')

    for _, row in full_data_df.iterrows():        
        x_idx = int(600 - np.round((np.abs(row['surf_x']) - surf_x_min) / 150) - 1)
        y_idx = int(np.round((np.abs(row['surf_y']) - surf_y_min) / 150))
        if 0 <= x_idx < 600 and 0 <= y_idx < 600:
            radar_mask[x_idx, y_idx] = True

    radar_mask_tensor = torch.tensor(radar_mask, dtype=torch.bool)

    dataset = BedTopoDataset(inputs, target_bed, radar_mask_tensor, patch_size=16, stride=8)
    model = BedTopoCNN(in_channels=inputs.shape[0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular'
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    patience = 5000  
    best_loss = float('inf')  
    counter = 0  
    num_epochs = 20000


    for epoch in range(num_epochs):        
        model.train()
        radar_loss_sum = 0.0
        bedmachine_loss_sum = 0.0
        batch_count = 0
    
        for input_patch, target_patch, radar_mask_patch in train_loader:
            input_patch = input_patch.to(device)
            target_patch = target_patch.to(device)
            radar_mask_patch = radar_mask_patch.to(device)

            optimizer.zero_grad()
            outputs = model(input_patch).squeeze(1)
            loss_radar, loss_bedmachine, loss = custom_loss(outputs, target_patch, radar_mask_patch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            radar_loss_sum += loss_radar.item()
            bedmachine_loss_sum += loss_bedmachine.item()
            batch_count += 1

        avg_radar_loss = radar_loss_sum / batch_count
        avg_bedmachine_loss = bedmachine_loss_sum / batch_count
        print(f"Epoch [{epoch+1}/{num_epochs}], Radar Loss: {avg_radar_loss:.4f}, BedMachine Loss: {avg_bedmachine_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_patch, target_patch, radar_mask_patch in val_loader:
                input_patch = input_patch.to(device)
                target_patch = target_patch.to(device)
                radar_mask_patch = radar_mask_patch.to(device)

                outputs = model(input_patch).squeeze(1)
                _, _, loss = custom_loss(outputs, target_patch, radar_mask_patch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Early stopping and model saving
        if avg_val_loss < best_loss:
            print(f"Validation loss improved from {best_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_loss = avg_val_loss
            torch.save(model.state_dict(), './saved_models/best_model_16_stride_8.pth')
            counter = 0  # Reset counter if improvement
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")

        if counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

if __name__ == "__main__":
    main()