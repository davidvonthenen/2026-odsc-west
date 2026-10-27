# Task 0: Attendee environment setup

**Workshop:** Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Solutions  
**Section:** Environment setup

Complete these steps from the workshop setup directory (scripts/0_setup) containing `setup.sh`, `stop.sh`, `cleanup.sh`, and `validate_environment.py`. Use the same Bash terminal for the API-key exports, container startup, and validation.

## 1. Check prerequisites

You need:

- **Python 3.10 or newer**, with `pip` and `venv`. If Python is missing, follow [Python installation and configuration](python-installation.md).
- **Podman** (or Docker), with a running Podman engine. Install Podman using the [official Podman instructions](https://podman.io/docs/installation).
- **Bash, Git, curl, OpenSSL, and make**, available in the terminal you use for the workshop.
- Internet connectivity for the initial setup, OpenAI API calls, and Tavily validation.

### macOS Instructions

Run the workshop commands in Terminal. Install Podman using the official macOS installer. If Git or make is missing, install the Xcode Command Line Tools:

```bash
xcode-select --install
```

Initialize and start the Podman machine if needed:

```bash
podman machine list
podman machine init       # Only if no machine exists
podman machine start      # Only if the machine is stopped
```

`python3 --version` must report **3.10 or newer**, and `podman info` must succeed.

### Linux (Debian/Ubuntu)

 Install the required host commands if they are missing:

```bash
sudo apt-get update
sudo apt-get install -y podman git curl openssl make
```

Verify prerequisites in your workshop Bash terminal:

```bash
python3 --version
command -v bash git curl openssl make
podman --version
podman info
```

`python3 --version` must report **3.10 or newer**, and `podman info` must succeed.

### Linux (Fedora)

Install the required host commands if they are missing:

```bash
sudo dnf install -y podman git curl openssl make
```

Verify prerequisites in your workshop Bash terminal:

```bash
python3 --version
command -v bash git curl openssl make
podman --version
podman info
```

`python3 --version` must report **3.10 or newer**, and `podman info` must succeed.

### Windows

**WARNING:** All Windows instructions in the lab are untested. Use at your own risk. 
**RECOMMENDATION:** It's recommended to use a VM or Cloud Instance with Linux. This will NOT be provided.

Run workshop commands in **Ubuntu on WSL2**, not in PowerShell. Install Podman for Windows and configure access to its engine from your WSL distribution using the [Podman Desktop WSL instructions](https://podman-desktop.io/docs/podman/accessing-podman-from-another-wsl-instance). In your Ubuntu terminal, install the host commands using the Debian/Ubuntu command above (if needed). For WSL setup, see [Microsoft's installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

Initialize and start the Podman machine if needed:

```bash
podman machine list
podman machine init       # Only if no machine exists
podman machine start      # Only if the machine is stopped
```

`python3 --version` must report **3.10 or newer**, and `podman info` must succeed.

### Set the OpenSearch kernel prerequisite

OpenSearch requires `vm.max_map_count` of at least `262144` on the Linux kernel running the containers. On **macOS** or **Windows**, run this command on the host (macOS Terminal or Windows PowerShell) against the Podman machine:

```bash
podman machine ssh "sudo sysctl -w vm.max_map_count=262144"
```

On **native Linux**, run:

```bash
sudo sysctl -w vm.max_map_count=262144
```

This setting may need to be applied again after restarting the host or Podman machine. See the [OpenSearch container requirements](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/).

## 2. git clone

Grab a copy of the GitHub repository, by running the following command in a new working directory:

```bash
mkdir -p ./mylab
cd ./mylab
git clone https://github.com/davidvonthenen/2026-odsc-west.git
```

Then proceed to the `scripts` directory in the repository you just downloaded:

```bash
cd 2026-odsc-west/workshop/0_setup
```

## 3. Prepare Python and API keys

In the workshop project directory, create or activate a Python virtual environment, then install the validator's Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Obtain an [OpenAI API key](https://platform.openai.com/api-keys) (or an OpenAI-compatible inference provider) and a [Tavily API key](https://app.tavily.com/). Both keys are required. No other environment-variable changes are needed with the supplied workshop defaults. If you don't have an OpenAI API Key, Nebius and Groq are highly recommended.

| Provider | API base URL | Provider documentation |
|---|---|---|
| Nebius | `https://api.tokenfactory.nebius.com/v1` | [OpenAI compatibility](https://docs.tokenfactory.nebius.com/quickstart) |
| Groq | `https://api.groq.com/openai/v1` | [OpenAI compatibility](https://console.groq.com/docs/openai) |
| Together AI | `https://api.together.ai/v1` | [OpenAI compatibility](https://docs.together.ai/docs/inference/openai-compatibility) |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` | [OpenAI compatibility](https://docs.fireworks.ai/tools-sdks/openai-compatibility) |

Enter the keys securely in the **same Bash terminal** from which you will run setup and validation:

```bash
read -r -s -p "OPENAI_API_KEY: " OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY

read -r -s -p "TAVILY_API_KEY: " TAVILY_API_KEY
printf '\n'
export TAVILY_API_KEY
```

These exports remain available in this terminal session. Repeat them in a new terminal session if you need to rerun setup or validation.

## 4. Start the workshop containers

From the workshop project directory, run:

```bash
./setup.sh
```

The script starts **five containers**:

| Service | Container | Local endpoint |
| --- | --- | --- |
| PostgreSQL with pgvector | `postgres` | `127.0.0.1:8432` |
| Mem0 API | `mem0` | `http://127.0.0.1:8888` |
| Mem0 dashboard | `mem0-dashboard` | `http://127.0.0.1:3000` |
| OpenSearch | `opensearch-single` | `http://127.0.0.1:9200` |
| OpenSearch Dashboards | `opensearch-single-dashboards` | `http://127.0.0.1:5601` |

The first startup downloads or builds required images and initializes the services. **If container downloads or builds are taking too long**, use the instructor-provided USB archives and [load container images from USB](load-images-from-usb.md), then return here and run `./setup.sh`.

## 5. Validate your environment

From the workshop project directory, with your Python environment active and both API keys exported:

```bash
python validate_environment.py
```

The validation checks the host tools, Podman, five containers, OpenSearch and k-NN, PostgreSQL and pgvector, Mem0, OpenAI chat and embeddings, and the Tavily key. OpenAI connectivity checks make small billable API requests. **Continue to the workshop only when validation reports zero failures.**

### Stopping, restarting, or cleaning up

**Stop containers without deleting workshop data:**

```bash
./stop.sh
```

**Restart after stopping:**

```bash
./setup.sh
```

The stop script removes the workshop containers and networks but leaves the PostgreSQL volume and OpenSearch data directory in place. Reusing `setup.sh` recreates the containers with those stored files.

**Clean up at the end of the workshop (destructive):**

```bash
./stop.sh
./cleanup.sh
```

`cleanup.sh` removes the Mem0 checkout, OpenSearch files under `~/opensearch`, and selected container images. **The supplied cleanup script does not remove the named PostgreSQL volume `postgres_db`.** Do not run cleanup when you intend to keep workshop data.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `podman info` fails | Start the Podman machine on macOS/Windows. On Windows, verify the WSL Podman connection. |
| Python package import fails | Activate `.venv` and rerun the package-install command in Step 2. |
| OpenSearch exits during startup | Verify the `vm.max_map_count` setting and inspect `podman logs opensearch-single`. |
| Setup stalls before OpenSearch starts | Inspect `podman logs --tail 100 postgres`, `mem0`, and `mem0-dashboard`. |
| OpenAI authentication, model, or quota fails | Check `OPENAI_API_KEY` and account access to the model specified by the workshop. |
| Tavily authentication or quota fails | Check `TAVILY_API_KEY` and Tavily account usage. |
| Container names are already in use | Use `./stop.sh` before restarting the workshop stack. |
