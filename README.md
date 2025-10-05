# NASA Space Apps Challenges 2025 - Hunting for Exoplanets using AI

## The `uv` Package Manager

[`uv`](https://github.com/astral-sh/uv) is a fast, modern Python package manager and environment tool built as a replacement for `pip` and `venv`.

This guide covers installation steps for **Windows** and **macOS**, setting up a virtual environment, and managing dependencies for your project.

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
   Ensure `uv.exe` is in your `PATH`. By default, it installs under:

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

### 🐍 Creating a Virtual Environment with `uv`

Once `uv` is installed, you can create and manage virtual environments seamlessly.

1. **Create a new project (optional):**

   ```bash
   uv init my-project
   cd my-project
   ```

   This sets up a new project directory with a `pyproject.toml`.

2. **Create a virtual environment:**

   ```bash
   uv venv
   ```

   By default, this creates a `.venv` folder inside your project.

3. **Activate the environment:**

   * **Windows (PowerShell):**

     ```powershell
     .venv\Scripts\Activate.ps1
     ```

   * **macOS/Linux (bash/zsh):**

     ```bash
     source .venv/bin/activate
     ```

---

### 📚 Adding Dependencies with `uv sync`

The `uv sync` command installs and synchronizes dependencies defined in `pyproject.toml`.

1. **Add a package (example: `numpy`):**

   ```bash
   uv add numpy
   ```

   This updates `pyproject.toml` and `uv.lock`.

2. **Install all dependencies:**

   ```bash
   uv sync
   ```

3. **Run Python inside the environment:**

   ```bash
   uv run python main.py
   ```

---

### 📑 Example `pyproject.toml`

For AI/ML projects like this NASA Space Apps challenge, a minimal `pyproject.toml` might look like:

```toml
[project]
name = "exoplanet-ai"
version = "0.1.0"
description = "AI models for exoplanet detection"
authors = [{ name = "Team Name" }]
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "tensorflow", # or "torch"
]
```

You can edit dependencies directly or add them using `uv add <package>`.

---

✅ At this point, you have:

* Installed `uv`
* Created and activated a virtual environment
* Added dependencies and synchronized them
* A reproducible setup for your NASA Space Apps challenge project
