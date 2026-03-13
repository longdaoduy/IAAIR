"""
IAAIR Monitoring — Google Colab Setup (Native Binaries)
=======================================================
Runs Prometheus + Grafana Alloy as native Linux binaries (no Docker needed).
Metrics are pushed to Grafana Cloud at:
  https://daongochoa2002.grafana.net

Why native instead of udocker?
  - udocker's proot mode breaks localhost networking
  - udocker doesn't reliably pass -e environment variables
  - Native binaries are simpler and just work on Colab
"""

import os
import time
import subprocess

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = "/content/IAAIR/monitoring"
BIN_DIR  = "/content/monitoring_bin"

# Grafana Cloud credentials — load from Colab Secrets (🔑 sidebar)
# Required secrets: GRAFANA_CLOUD_PROM_USERNAME, GRAFANA_CLOUD_API_KEY,
#                   GRAFANA_CLOUD_REMOTE_WRITE_URL
try:
    from google.colab import userdata
    GRAFANA_CLOUD_PROM_USERNAME  = userdata.get("GRAFANA_CLOUD_PROM_USERNAME")
    GRAFANA_CLOUD_API_KEY        = userdata.get("GRAFANA_CLOUD_API_KEY")
    GRAFANA_CLOUD_REMOTE_WRITE_URL = userdata.get("GRAFANA_CLOUD_REMOTE_WRITE_URL")
    print("   🔑 Loaded credentials from Colab Secrets")
except Exception:
    GRAFANA_CLOUD_PROM_USERNAME  = os.environ.get("GRAFANA_CLOUD_PROM_USERNAME", "")
    GRAFANA_CLOUD_API_KEY        = os.environ.get("GRAFANA_CLOUD_API_KEY", "")
    GRAFANA_CLOUD_REMOTE_WRITE_URL = os.environ.get("GRAFANA_CLOUD_REMOTE_WRITE_URL", "")
    print("   📋 Loaded credentials from environment variables")

if not GRAFANA_CLOUD_API_KEY:
    raise ValueError(
        "❌ GRAFANA_CLOUD_API_KEY not set!\n"
        "   Add it as a Colab Secret (🔑) or set it as an environment variable.\n"
        "   Required secrets: GRAFANA_CLOUD_PROM_USERNAME, GRAFANA_CLOUD_API_KEY, "
        "GRAFANA_CLOUD_REMOTE_WRITE_URL"
    )

# IAAIR API target — detect host IP for udocker/container compatibility
# Inside udocker containers, "localhost" doesn't resolve to the host.
# We use the host's actual IP address instead.
import socket
def _get_host_ip():
    """Get the host machine's IP (works on Colab and local)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "172.28.0.12"  # Colab default host IP

HOST_IP = _get_host_ip()
API_TARGET = f"{HOST_IP}:8000"
print(f"   🌐 Host IP: {HOST_IP} → API target: {API_TARGET}")

# ── 1. Cleanup ─────────────────────────────────────────────────
print("🛑 Cleaning up old processes...")
os.system("pkill -f 'prometheus.*config' 2>/dev/null || true")
os.system("pkill -f 'alloy.*run' 2>/dev/null || true")
os.system("fuser -k 9090/tcp 12345/tcp 2>/dev/null || true")
time.sleep(2)

# ── 2. Download binaries (once) ────────────────────────────────
os.makedirs(BIN_DIR, exist_ok=True)

PROM_VERSION = "2.51.2"
ALLOY_VERSION = "1.5.1"

prom_bin  = f"{BIN_DIR}/prometheus"
alloy_bin = f"{BIN_DIR}/alloy"

if not os.path.isfile(prom_bin):
    print(f"📥 Downloading Prometheus v{PROM_VERSION}...")
    prom_tar = f"prometheus-{PROM_VERSION}.linux-amd64"
    os.system(
        f"cd {BIN_DIR} && "
        f"curl -sSL https://github.com/prometheus/prometheus/releases/download/"
        f"v{PROM_VERSION}/{prom_tar}.tar.gz | tar xz && "
        f"mv {prom_tar}/prometheus . && mv {prom_tar}/promtool . && "
        f"rm -rf {prom_tar}"
    )
    print("   ✅ Prometheus downloaded")
else:
    print("   ✅ Prometheus already present")

if not os.path.isfile(alloy_bin):
    print(f"📥 Downloading Grafana Alloy v{ALLOY_VERSION}...")
    alloy_zip = f"alloy-linux-amd64"
    os.system(
        f"cd {BIN_DIR} && "
        f"curl -sSL https://github.com/grafana/alloy/releases/download/"
        f"v{ALLOY_VERSION}/{alloy_zip}.zip -o alloy.zip && "
        f"unzip -o alloy.zip && mv {alloy_zip} alloy && rm alloy.zip"
    )
    print("   ✅ Alloy downloaded")
else:
    print("   ✅ Alloy already present")

# ── 3. Render prometheus.yml (expand placeholders) ─────────────
print("📝 Rendering prometheus.yml with credentials...")
os.makedirs(f"{BIN_DIR}/data", exist_ok=True)

rendered_prom_yml = f"{BIN_DIR}/prometheus.yml"
with open(f"{BASE_DIR}/prometheus.yml", "r") as f:
    template = f.read()

rendered = (
    template
    .replace("${GRAFANA_CLOUD_PROM_USERNAME}", GRAFANA_CLOUD_PROM_USERNAME)
    .replace("${GRAFANA_CLOUD_API_KEY}", GRAFANA_CLOUD_API_KEY)
    .replace("${GRAFANA_CLOUD_REMOTE_WRITE_URL}", GRAFANA_CLOUD_REMOTE_WRITE_URL)
    .replace("localhost:8000", API_TARGET)
)
with open(rendered_prom_yml, "w") as f:
    f.write(rendered)
print(f"   ✅ Written to {rendered_prom_yml}")

# ── 4. Render alloy-config.alloy (replace env() with literals) ─
print("📝 Rendering alloy-config.alloy with credentials...")
rendered_alloy_cfg = f"{BIN_DIR}/config.alloy"
with open(f"{BASE_DIR}/alloy-config.alloy", "r") as f:
    alloy_template = f.read()

# Replace env("VAR_NAME") calls with literal quoted values
rendered_alloy = (
    alloy_template
    .replace('env("GRAFANA_CLOUD_REMOTE_WRITE_URL")', f'"{GRAFANA_CLOUD_REMOTE_WRITE_URL}"')
    .replace('env("GRAFANA_CLOUD_PROM_USERNAME")', f'"{GRAFANA_CLOUD_PROM_USERNAME}"')
    .replace('env("GRAFANA_CLOUD_API_KEY")', f'"{GRAFANA_CLOUD_API_KEY}"')
    .replace("localhost:8000", API_TARGET)
)
with open(rendered_alloy_cfg, "w") as f:
    f.write(rendered_alloy)
print(f"   ✅ Written to {rendered_alloy_cfg}")

# ── 5. Start Prometheus ────────────────────────────────────────
print("🚀 Starting Prometheus (port 9090)...")
prom_log = "/content/prometheus.log"
prom_proc = subprocess.Popen(
    [
        prom_bin,
        f"--config.file={rendered_prom_yml}",
        f"--storage.tsdb.path={BIN_DIR}/data",
        "--storage.tsdb.retention.time=24h",
        "--web.listen-address=0.0.0.0:9090",
        "--web.enable-lifecycle",
    ],
    stdout=open(prom_log, "w"),
    stderr=subprocess.STDOUT,
)
print(f"   PID: {prom_proc.pid}")

# ── 6. Start Grafana Alloy ────────────────────────────────────
print("🔄 Starting Grafana Alloy (port 12345)...")
alloy_log = "/content/alloy.log"
alloy_proc = subprocess.Popen(
    [
        alloy_bin,
        "run",
        rendered_alloy_cfg,
        "--server.http.listen-addr=0.0.0.0:12345",
    ],
    stdout=open(alloy_log, "w"),
    stderr=subprocess.STDOUT,
)
print(f"   PID: {alloy_proc.pid}")

# ── 7. Health Checks ──────────────────────────────────────────
print("\n⏳ Waiting 8s for processes to start...")
time.sleep(8)

# Check if processes are still running
prom_alive  = prom_proc.poll() is None
alloy_alive = alloy_proc.poll() is None

print(f"\n── Process Status ──")
print(f"   Prometheus:  {'✅ running' if prom_alive else '❌ exited (code ' + str(prom_proc.returncode) + ')'}")
print(f"   Alloy:       {'✅ running' if alloy_alive else '❌ exited (code ' + str(alloy_proc.returncode) + ')'}")

print(f"\n── Prometheus Log (last 5 lines) ──")
os.system(f"tail -n 5 {prom_log}")

print(f"\n── Alloy Log (last 5 lines) ──")
os.system(f"tail -n 5 {alloy_log}")

print(f"\n── Prometheus Targets ──")
os.system(
    "curl -s http://localhost:9090/api/v1/targets 2>/dev/null "
    "| python3 -c \"import sys,json; d=json.load(sys.stdin); "
    "[print(f'  {t[\\\"labels\\\"][\\\"job\\\"]:20s} {t[\\\"health\\\"]:6s}  {t[\\\"lastScrape\\\"]}') "
    "for t in d.get(\\\"data\\\",{}).get(\\\"activeTargets\\\",[])]\" "
    "2>/dev/null || echo '  (not ready yet — check log above)'"
)

print(f"\n{'=' * 60}")
print(f"✅ Monitoring started!")
print(f"   Prometheus:    http://localhost:9090")
print(f"   Alloy UI:      http://localhost:12345")
print(f"   Grafana Cloud: https://daongochoa2002.grafana.net")
print(f"{'=' * 60}")
