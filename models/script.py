import torch
import torch.nn as nn
import time
import sys

# --- RE-DEFINE THE CLASS ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, (hidden, cell) = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        return dec_out

# ---------------------------------------------------------

print("--- STARTING CONVERSION PROCESS ---")

# 1. Load Model
print("[1/4] Initializing model architecture...", end=" ")
model = LSTMAutoencoder()
print("Done.")

print(f"[2/4] Loading weights from 'LSTM_NSR_autoencoder_10s.pth'...", end=" ")
try:
    model.load_state_dict(torch.load("LSTM_NSR_autoencoder_10s.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Done.")
except FileNotFoundError:
    print("\nERROR: Could not find 'LSTM_NSR_autoencoder_10s.pth'. Check the filename!")
    sys.exit(1)

# 2. Create Dummy Input
print("[3/4] Creating dummy input tensor (1, 2500, 1)...", end=" ")
dummy_input = torch.randn(1, 2500, 1)
print("Done.")

# 3. Export to ONNX
output_path = "ecg_model.onnx"
print(f"[4/4] Exporting to ONNX (This may take a moment)...")

start_time = time.time()

# We can't put a progress bar INSIDE this function, but this print confirms it started

torch.onnx.export(
    model, 
    dummy_input, 
    output_path,
    export_params=True,
    opset_version=14,         # UPDATED: Changed from 11 to 14 to satisfy newer PyTorch
    do_constant_folding=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
)

end_time = time.time()
duration = end_time - start_time

print(f"\nSUCCESS! Model saved to '{output_path}'")
print(f"Time taken: {duration:.2f} seconds")