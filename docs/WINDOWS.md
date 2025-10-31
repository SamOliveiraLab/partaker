# Windows Quickstart (CPU-only, No GPU)

This guide is for Windows users running **nd2-timelapse-analyzer** (partaker) on a machine without a CUDA GPU.  
The steps here help you avoid common Windows/Python/PySide6 pitfalls and ensure TensorFlow works for segmentation.

---

## 1. Install Python and Dependencies

- Use **Anaconda/Miniconda** (recommended for best results).
- Create and activate your environment:
    ```sh
    conda create -n partaker python=3.10
    conda activate partaker
    ```

- Install with [uv](https://github.com/astral-sh/uv) (or pip, but uv is recommended for lockfile support).

- **Check your `pyproject.toml`** for this block in the `[project].dependencies` list:
    ```toml
    "pyside6 (==6.7.2)",
    # --- >>> PLATFORM-SPECIFIC TENSORFLOW (THIS BLOCK IS THE FIX) <<<
    # Windows, Linux: regular tensorflow
    "tensorflow (>=2.13.0,<2.16.0) ; platform_system != 'Darwin' and platform_system != 'Windows'",
    # Mac ARM (Apple Silicon)
    "tensorflow-macos (==2.15) ; platform_system == 'Darwin' and platform_machine == 'arm64'",
    # Mac Intel (if you ever use Intel Mac)
    "tensorflow-macos (==2.15) ; platform_system == 'Darwin' and platform_machine != 'arm64'",
    # Windows: Intel/AMD optimized
    "tensorflow-intel (>=2.13.0,<2.16.0) ; platform_system == 'Windows'",
    # Tensorflow IO GCS filesystem for non-Mac-ARM
    "tensorflow-io-gcs-filesystem (>=0.23.1) ; platform_machine != 'arm64' or platform_system != 'Darwin'",
    # (Old) Windows: keep compatibility version
    "tensorflow-io-gcs-filesystem (<0.32.0) ; platform_system == 'Windows'",
    # --- >>> END OF PLATFORM-SPECIFIC BLOCK <<<
    "polars (>=1.30.0,<2.0.0)",
    ```

- Then install dependencies:
    ```sh
    uv pip install -r pyproject.toml
    # or, if not using uv:
    pip install -r requirements.txt
    ```

---

## 2. Launching the GUI on Windows

**Do NOT run the app with `python src/nd2_analyzer/ui/app.py`**, as this can cause import errors.

Instead, from the `partaker\src` folder, run:

```sh
python -u -m nd2_analyzer.ui.app
```

- The `-u` flag makes sure all terminal output (info, warnings, errors) is visible in real time.
- This is different from Mac/Linux, where you can simply run:
    ```sh
    uv run gui
    ```

---

## 3. Why This Is Needed

- Windows needs `"tensorflow-intel"` for CPU-only TensorFlow support.
- The modular launch style (`python -u -m ...`) is required for Python to find and import the `nd2_analyzer` package correctly on Windows.
- Mac users can still run via `uv run gui`, and all platform-specific requirements are handled in the `pyproject.toml`—so these instructions do **not** break Mac/Linux compatibility.

---

## 4. Example: Main Entrypoint

Make sure the bottom of your `partaker\src\nd2_analyzer\ui\app.py` includes:

```python
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    from nd2_analyzer.ui.app import App  # or MainWindow, depending on your code
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
```

---

## 5. Troubleshooting

- If you see `ModuleNotFoundError: No module named 'nd2_analyzer'`, make sure you are running **from the `src` folder** and using the `-m` flag.
- If you see TensorFlow errors, double-check that `"tensorflow-intel"` is installed and that your conda environment is active.

### 5.a DataParallel `load_model` error (Windows CPU)
If segmentation fails with an error like:
```
AttributeError: 'DataParallel' object has no attribute 'load_model'
```
this can happen with some Omnipose/Cellpose versions when running on CPU (Windows). Before patching, first try upgrading Omnipose:

```sh
pip install --upgrade omnipose
```

If the error persists and you need a quick workaround, edit the installed Omnipose file (example path — adjust for your environment):

`<python-env>/Lib/site-packages/cellpose_omni/core.py`

Find `_run_nets` and replace the direct `load_model` call with a safe check that uses `.module` if present. Example minimal change:

```python
# before (problematic)
net.load_model(self.pretrained_model[0], cpu=(not self.gpu))

# after (safe)
if hasattr(net, 'module'):
    net.module.load_model(self.pretrained_model[0], cpu=(not self.gpu))
else:
    net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
```

Notes:
- This is a local, temporary patch in `site-packages` and will be overwritten if you upgrade/reinstall Omnipose.
- Prefer filing a PR or using an upstream fix; upgrade first before applying this patch.
- After editing, restart your Python session and re-run the app.

---

## 6. For Mac/Linux Users

- The above steps are **not** required for Mac/Linux.
- On Mac/Linux, simply run:
    ```sh
    uv run gui
    ```
- All dependencies and entrypoints are automatically handled.

---

## 7. Questions?

- If you hit issues not covered here, open an issue on the repo or ask the maintainers.

---