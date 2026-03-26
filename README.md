# 🎭 DeepFake Detector

An advanced deepfake detection tool powered by a custom pretrained model (`ig.bin`), built with PyTorch and Gradio. Supports both a web-based UI and a command-line interface for analyzing videos.

---

## 🚀 Features

- 🔍 **Face-level analysis** — detects and extracts facial regions before classification
- 🎬 **Multi-frame sampling** — analyzes ~15 frames per video for robust results
- ⚖️ **Confidence scoring** — weighted average across frames for reliable verdicts
- 🖥️ **Web UI** — clean Gradio interface for non-technical users
- ⌨️ **CLI mode** — run detection directly from the terminal
- ⚡ **GPU acceleration** — automatic CUDA detection and usage
- 📄 **Auto export** — saves results to a `.txt` file after each CLI run

---

## 📦 Requirements

```bash
pip install torch torchvision gradio opencv-python pillow numpy
```

> **Python 3.8+** is recommended.  
> For GPU support, install the appropriate [PyTorch CUDA build](https://pytorch.org/get-started/locally/).

---

## 🧠 Model — `ig.bin`

The model file (`ig.bin`) is **not included** in this repository due to its size.

### ⬇️ Download from Hugging Face

```
https://huggingface.co/accel69/depfake-detection/resolve/main/ig.bin
```

> Replace `<your-username>` and `<your-repo>` with your actual Hugging Face details.

After downloading, place `ig.bin` in the **same directory** as `app_best.py`:

```
your-project/
├── app_best.py
├── ig.bin          ← place here
└── README.md
```

---

## ▶️ Usage

### 🌐 Web Interface

```bash
python app_best.py
```

- Opens automatically in your browser at `http://127.0.0.1:7860`
- Upload a video file and click **Analyze Video**
- View the verdict, probability scores, and confidence level

**Custom port:**
```bash
GRADIO_SERVER_PORT=7861 python app_best.py
```

---

### ⌨️ CLI Mode

```bash
python app_best.py <path_to_video>
```

**Examples:**
```bash
python app_best.py video.mp4
python app_best.py /path/to/test_clip.avi
```

---

## 📊 Understanding Results

| Verdict | Meaning |
|---|---|
| ✅ **REAL VIDEO** | High confidence the video is genuine |
| 🚨 **FAKE DETECTED** | Deepfake or facial manipulation artifacts found |
| ⚠️ **UNCERTAIN** | Model confidence too low — manual review recommended |

### Confidence Score Guide

| Range | Reliability |
|---|---|
| 0–50% | Low — result may be unreliable |
| 50–75% | Medium — reasonably reliable |
| 75–100% | High — very reliable result |

---

## 🔬 How It Works

1. **Frame Sampling** — Extracts ~15 evenly spaced frames from the video
2. **Face Detection** — Uses OpenCV Haar Cascade to locate facial regions
3. **Preprocessing** — Resizes faces to 224×224, normalizes with ImageNet stats
4. **Inference** — Runs each face through the `ig.bin` model
5. **Aggregation** — Weighted average of per-frame predictions (weighted by confidence)
6. **Verdict** — Final classification based on aggregated scores

---

## 🗂️ Project Structure

```
your-project/
├── app_best.py       # Main application (web + CLI)
├── ig.bin            # Pretrained model weights (download separately)
└── README.md
```

---

## ⚠️ Limitations

- Best results on videos with **clear, well-lit facial regions**
- Accuracy varies with **video quality** and **deepfake sophistication**
- Not intended as a forensic tool — designed to **flag suspicious content**
- Minimum recommended face size: **80×80 pixels**

---

## 💻 Hardware

| Setup | Speed |
|---|---|
| NVIDIA GPU (CUDA) | Fast (~seconds per video) |
| CPU only | Slower (~minutes per video) |

The app auto-detects CUDA and uses GPU when available.

---

## 📝 License

This project is released for research and educational purposes.  
Please use responsibly and do not deploy for surveillance or harassment.

---

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) — deep learning framework  
- [Gradio](https://www.gradio.app/) — web UI  
- [OpenCV](https://opencv.org/) — face detection  
- [Hugging Face](https://huggingface.co/) — model hosting
