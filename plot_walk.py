import subprocess
import re
import matplotlib.pyplot as plt
import os

print("Running angad_smooth_walk.py to collect data...")
result = subprocess.run(
    ["/home/charan/venvs/mujoco-humble/bin/python3", "angad_smooth_walk.py", "--headless"],
    capture_output=True, text=True
)

output = result.stdout
times = []
fwds = []
rolls = []

for line in output.split('\n'):
    # Example line: "    1.5s │   0   │  -2.7°   -1.4° │    +0.2cm │ ✓ STABLE"
    match = re.search(r'^\s*([\d\.]+)s\s*│\s*\d+\s*│\s*([+-][\d\.]+)°\s*[+-][\d\.]+°\s*│\s*([+-][\d\.]+)cm', line)
    if match:
        times.append(float(match.group(1)))
        rolls.append(float(match.group(2)))
        fwds.append(float(match.group(3)))

if not times:
    print("No data parsed!")
    print(output)
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(times, fwds, label="Forward Distance (cm)", color="blue", linewidth=2)
plt.plot(times, rolls, label="Lateral Roll (deg)", color="red", alpha=0.5)
plt.title("Angad Walking Trajectory (Virtual ZMP Stabilizer)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.grid(True)
plt.legend()

save_path = "/home/charan/.gemini/antigravity/brain/ac70fb58-8c3a-4034-901c-119921606efa/graphify_visualization.png"
plt.savefig(save_path)
print(f"Graph saved to {save_path}")
