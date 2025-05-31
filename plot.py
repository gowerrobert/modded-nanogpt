import re
import matplotlib.pyplot as plt
import os
import numpy as np
# Replace with the actual paths to your log files
gpu4_logs = {
    # "PolarExpress0": "logs/polar-def-0.txt",
    # "PolarExpress01": "logs/polar-def-0-1.txt",
    # "PolarExpress001": "logs/polar-0-01.txt",
    # "NewtonSchulz5": "logs/Newton5.txt",
    # "PolarExpress001-ns-4": "logs/ns-4-polar.txt",
    # "PolarExpress001-rep": "logs/polar-0-001-4k-repeat.txt",
    # "PolarExpress001": "logs/polar-0-001-4k.txt",
    # "NewtonSchulz5": "logs/Newton5-4k.txt"   
}
gpu8_logs = {
    # "NewtonSchulz5": "logs/Newton58GPU.txt",
    # "PolarExpress": "logs/polar-8GPU.txt"
    # "NewtonSchulz5": "logs/Newton5-2k-iter-8GPU.txt",
    # "PolarExpress": "logs/Polar-2k-iter-8GPU.txt" 
#    "NewtonSchulz5": "logs/Newton5-2.5k-8GPU.txt",
#    "PolarExpress": "logs/Polar-2.5k-8GPU.txt" 
   "NewtonSchulz5": "logs/med-Newton5-fixedp.txt",
#    "PolarExpress": "logs/med-Polar-10min7.txt" 
   "PolarExpress":  "logs/med-Polar-defl-pow.txt"
}
# Regex patterns to extract step, validation loss, and time
patterns = [
    r"step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms"  # Additional pattern
]

# Function to process log files and extract data
def process_logs(log_files, batch_size_multiplier):
    output = {}
    for name, log_file in log_files.items():
        steps = []
        val_losses = []
        times = []
        with open(log_file, "r") as f:
            for line in f:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        step = int(match.group(1)) * batch_size_multiplier
                        if "val_loss" in pattern:
                            val_loss = float(match.group(2))
                            time = float(match.group(3))
                        else:
                            val_loss = None  # Placeholder if val_loss is not in the pattern
                            time = float(match.group(2))
                        steps.append(step)
                        val_losses.append(val_loss)
                        times.append(time)
                        break
        output[name] = {"steps": steps, "losses": val_losses, "times": times}
    return output
# Batch size multipliers
val_loss_skip = 125
batchsize_8 = 8 * 48 * 1024 * val_loss_skip
batchsize_4 = 4 * 24 * 1024 * val_loss_skip

# Process GPU4 and GPU8 logs
gpu4_data = process_logs(gpu4_logs, batchsize_4)
gpu8_data = process_logs(gpu8_logs, batchsize_8)

# Combine data
all_data = {**gpu4_data, **gpu8_data}
# Plot all times in one plot

plt.figure(figsize=(8, 5))
ymin =np.inf
for name, data in all_data.items():
    print(name +" final loss " +  str(data["losses"][-1]) + " time " + f"{data['times'][-1]/1000/60:.4g}" +  " min")
    plt.plot(data["times"], data["losses"], label=name, linewidth=3, alpha=0.8)
    ymin = min(ymin, min(data["losses"]))
plt.xlabel("Time")
plt.ylabel("Validation Loss")
# plt.title("Time Over Training Steps")
plt.ylim(ymin*0.95 , 5)  # Set maximum y value to 5
plt.legend()
# plt.grid()
plt.savefig("img/val_time.png")
plt.show()

# Plot all losses in another plot
plt.figure(figsize=(8, 5))
for name, data in all_data.items():
    plt.plot(data["steps"], data["losses"], label=name, linewidth=3, alpha=0.8)
plt.xlabel("Tokens Processed")
plt.ylabel("Validation Loss")
# plt.title("Validation Loss Over Training Steps")
plt.ylim(ymin*0.95 , 5)  # Set maximum y value to 5
plt.legend()
# plt.grid()
plt.savefig("img/val_epoch.png")
plt.show()
