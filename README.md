# NASA.Hunting-for-Exoplanets-with-AI


# Setup
## Installing `uv` Package Manager

[`uv`](https://github.com/astral-sh/uv) is a fast, modern Python package manager and environment tool built as a replacement for `pip` and `venv`.

This guide covers installation steps for **Windows** and **macOS**.

---

### 📦 Installation

#### Windows

1. **Using PowerShell (recommended):**

   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Verify installation:**

   ```powershell
   uv --version
   ```

3. **Add to PATH (if required):**

   * Ensure `uv.exe` is in your `PATH`. By default, it installs under:

     ```
     %USERPROFILE%\.cargo\bin
     ```

---

#### macOS

1. **Using Homebrew (recommended):**

   ```bash
   brew install uv
   ```

2. **Using the install script (alternative):**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Verify installation:**

   ```bash
   uv --version
   ```

---

### 🔧 Post-install Setup

* Ensure your terminal recognizes `uv`. If not, add the binary location to your shell configuration (`~/.zshrc`, `~/.bashrc`, or `PowerShell Profile`).
* Run:

  ```bash
  uv help
  ```

  to explore available commands.

---

### ✅ Next Steps

* Replace `pip install ...` with:

  ```bash
  uv add package_name
  ```
* Create and manage environments with:

  ```bash
  uv venv
  ```

For more details, check the [uv documentation](https://docs.astral.sh/uv/).
