# Python installation and configuration

**Recommended Python version: 3.12.x** (use the latest available patch release in the 3.12 series). The workshop's `validate_environment.py` supports Python **3.10 or newer**, but Python **3.12** is the recommended version for a consistent workshop environment.

Use this guide only if Python 3.12 is missing or you need to create a workshop-specific environment. Install Python in the **same operating-system environment** where you will run the workshop. On Windows, this means **Ubuntu under WSL2**, not Windows-native Python.

The installation steps below use [uv](https://docs.astral.sh/uv/), a Python version manager. It installs Python 3.12 without replacing your operating system's Python interpreter. `uv` is needed for setup only; the workshop continues to run with `python` and `pip`.

## macOS

1. Open Terminal and install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. Install Python 3.12 and verify the installation:

   ```bash
   uv python install 3.12
   python3.12 --version
   ```

   The reported version should begin with `Python 3.12.`.

3. Continue to [Create the workshop Python environment](#create-the-workshop-python-environment).

Do not remove or replace the Python installation managed by macOS. For an alternative installer, see the [Python.org macOS downloads](https://www.python.org/downloads/macos/); select an installer specifically from the **Python 3.12** series. Note that Python.org's final 3.12 macOS binary installer is **3.12.10**; later security releases are source-only. The `uv` method above can install a newer 3.12 patch release.

## Linux

Install `curl` and CA certificates if they are not available.

**Debian / Ubuntu:**

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates
```

**Fedora:**

```bash
sudo dnf install -y curl ca-certificates
```

Then, in your Linux Bash terminal, install Python 3.12:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
python3.12 --version
```

The reported version should begin with `Python 3.12.`. Continue to [Create the workshop Python environment](#create-the-workshop-python-environment).

You can also use your distribution's `python3.12` package if one is available. Distribution defaults differ, so installing a package named `python3` does **not** guarantee Python 3.12. Do not replace the Python interpreter used by the Linux package manager.

References: [uv Python installation](https://docs.astral.sh/uv/guides/install-python/) and [Python virtual environments](https://docs.python.org/3.12/tutorial/venv.html).

## Windows (WSL2 workshop environment)

1. If Ubuntu under WSL2 is not installed, open **Windows PowerShell as Administrator** and run:

   ```powershell
   wsl --install -d Ubuntu
   ```

2. Restart Windows if requested. Open the **Ubuntu terminal** and complete its initial account setup.

3. In **Ubuntu Bash**, install Python 3.12:

   ```bash
   sudo apt-get update
   sudo apt-get install -y curl ca-certificates
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   uv python install 3.12
   python3.12 --version
   ```

4. Verify that the version begins with `Python 3.12.`. Store the workshop project and its `.venv` under your WSL Linux home directory (for example, `~/workshop`), rather than on a Windows-mounted drive.

5. Continue to [Create the workshop Python environment](#create-the-workshop-python-environment) **inside Ubuntu**, not PowerShell.

For Podman connectivity from Ubuntu, see the [Podman Desktop WSL guide](https://podman-desktop.io/docs/podman/accessing-podman-from-another-wsl-instance).

References: [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and [Set up a WSL development environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment).

### Optional: Windows-native Python (not used for the workshop)

For independent Windows projects, install the [Python install manager](https://www.python.org/downloads/windows/) and select Python 3.12 from PowerShell:

```powershell
py install 3.12
py -3.12 --version
```

A Windows-native installation is separate from the Python installation inside WSL2. The workshop commands must use the **WSL2 installation**.

Reference: [Python on Windows](https://docs.python.org/3/using/windows.html).

## Create the workshop Python environment

From a **Bash terminal** in the workshop project directory, create a `.venv` using Python 3.12 and include `pip`:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python --version
python3 --version
python -m pip --version
```

Both Python version commands should report **Python 3.12.x**. If you installed Python 3.12 without `uv`, you may use `python3.12 -m venv .venv` instead of the `uv venv` command.

Install the workshop validation dependencies:

```bash
python -m pip install python-dotenv httpx openai "psycopg[binary]"
```

In each new Bash terminal, reactivate the virtual environment before running workshop Python commands:

```bash
source .venv/bin/activate
python --version
```

Keep the environment activated when returning to the main guide. This ensures that its `python3` commands use Python **3.12** rather than a different system installation.

Return to [Attendee environment setup](0-attendee-setup.md).
