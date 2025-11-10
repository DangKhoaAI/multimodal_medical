# Medical Multimodal AI: X-ray and Text Analysis

A comprehensive deep learning system that combines medical imaging (X-ray) and clinical text data for improved diagnostic accuracy. This project implements multimodal neural networks with explainable AI (XAI) capabilities for medical diagnosis.

##  Project Overview

This project develops a multimodal AI system that:
- Analyzes medical X-ray images using DenseNet121 (pre-trained on ChestX-ray14)
- Processes clinical text reports using CNN-based text classification
- Combines both modalities for enhanced diagnostic predictions
- Provides explainable AI visualizations using Grad-CAM and Integrated Gradients

##  Architecture

### Multimodal Model Components

1. **Image Branch**: DenseNet121 backbone with ChestX-ray14 pre-trained weights
2. **Text Branch**: CNN with multiple filter sizes for text feature extraction
3. **Fusion Layer**: Concatenation and dense layers for multimodal integration

### Key Features

- **Explainable AI**: Grad-CAM for image explanations, Integrated Gradients for text
- **Web Interface**: Flask-based interactive application
- **Multiple Inference Modes**: Text-only, image-only, and multimodal predictions
- **Real-time Explanations**: Visual heatmaps and highlighted text importance

##  Project Structure

```
medical-multimodal/
├── app.py                     # Flask web application
├── config.py                  # Configuration parameters
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
├── checkpoints/              # Trained model weights
│   ├── img_model_final.h5
│   ├── text_model_final.h5
│   └── multi_model.h5
├── data/                     # Dataset and preprocessing
│   ├── loader.py
│   ├── ids_raw_texts_labels.csv
│   └── processed data files
├── models/                   # Model architectures
│   ├── multimodalingRE.py   # Main multimodal model
│   ├── load_models.py       # Model loading utilities
│   └── cnn_model.py         # CNN implementations
├── explainability/          # XAI components
│   ├── image_explainer.py   # Grad-CAM for images
│   └── text_explainer.py    # Integrated Gradients for text
├── preprocessing/           # Data preprocessing notebooks
├── evaluation/             # Model evaluation scripts
├── static/                 # Web assets (CSS, images)
└── templates/             # HTML templates
```

##  Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd medical-multimodal
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models**
   - Place model weights in the `checkpoints/` directory
   - Ensure the following files exist:
     - `img_model_final.h5`
     - `text_model_final.h5`
     - `multi_model.h5`

4. **Prepare data**
   - Place your dataset in the `data/` directory
   - Ensure `ids_raw_texts_labels.csv` contains your text data

### Running the Application

#### Local Development
```bash
python app.py
```
The application will be available at `http://localhost:5000`

#### Docker Deployment
```bash
docker build -t medical-multimodal .
docker run -p 5000:5000 medical-multimodal
```

##  Usage

### Web Interface

The application provides three main tabs:

1. **AI in Healthcare**: Information about the project and AI in medical diagnosis
2. **XAI (Explainable AI)**: Educational content about explainable AI techniques
3. **Model Inference**: Interactive prediction interface with three modes:
   - **Text Only**: Analyze clinical reports
   - **Image Only**: Analyze X-ray images
   - **Multimodal**: Combined text and image analysis

### API Usage

The application accepts:
- **Text input**: Clinical reports or medical descriptions
- **Image input**: X-ray images (JPEG, PNG formats)
- **Combined input**: Both text and image for multimodal analysis

### Prediction Output

Each prediction provides:
- **Probability score**: Likelihood of abnormality (0-1 scale)
- **Visual explanations**: 
  - Grad-CAM heatmaps for images
  - Word importance highlighting for text
- **Interactive visualizations**: Plotly-based charts for text explanations

##  Configuration

Key parameters in `config.py`:

```python
# Model paths
IMG_MODEL_PATH = 'checkpoints/img_model_final.h5'
TXT_MODEL_PATH = 'checkpoints/text_model_final.h5'
MRG_MODEL_PATH = 'checkpoints/multi_model.h5'

# Image processing
IMG_TARGET_SIZE = (224, 224)

# Text processing
MAX_NUM_WORDS = 15000
MAX_SEQ_LENGTH = 140

# Explainability
IG_GRADIENT_THRESHOLD = 0.70
```

##  Model Details

### Text Model
- **Architecture**: Multi-filter CNN with embedding layer
- **Input**: Tokenized text sequences (max length: 140)
- **Vocabulary**: 15,000 most frequent words
- **Filters**: Multiple kernel sizes for n-gram feature extraction

### Image Model
- **Base Architecture**: DenseNet121
- **Pre-training**: ChestX-ray14 dataset
- **Input Size**: 224×224 RGB images
- **Feature Extraction**: Average pooling layer output

### Multimodal Fusion
- **Strategy**: Late fusion with concatenation
- **Architecture**: Dense layers with dropout and batch normalization
- **Output**: Single probability score for abnormality detection

##  Explainability Features

### Image Explanations (Grad-CAM)
- **Heatmap Generation**: Highlights important image regions
- **Contour Detection**: Outlines significant areas
- **Color Mapping**: VIRIDIS colormap for better visualization
- **Overlay Options**: Adjustable transparency and gamma correction

### Text Explanations (Integrated Gradients)
- **Word Importance**: Scores individual word contributions
- **Gradient Computation**: Baseline integration for attribution
- **Interactive Plots**: Plotly-based importance visualization
- **Threshold Filtering**: Configurable importance thresholds

##  Development

### Adding New Models
1. Implement model architecture in `models/`
2. Update `load_models.py` for model loading
3. Add explainability methods in `explainability/`
4. Update configuration in `config.py`

### Extending Explainability
- Modify `image_explainer.py` for new image XAI methods
- Extend `text_explainer.py` for additional text explanations
- Update visualization templates in `templates/`

### Custom Datasets
1. Prepare data in the required CSV format
2. Update preprocessing scripts in `preprocessing/`
3. Modify data loading in `data/loader.py`
4. Retrain models using scripts in `scripts/`

##  Requirements

### Core Dependencies
- **TensorFlow/Keras**: Deep learning framework
- **Flask**: Web application framework
- **OpenCV**: Image processing
- **Plotly**: Interactive visualizations
- **PIL/Pillow**: Image handling
- **NumPy/Pandas**: Data manipulation

### Explainability Libraries
- **tf-keras-vis**: Grad-CAM implementation
- **Integrated Gradients**: Custom implementation included

##  Docker Support

The project includes Docker configuration for easy deployment:

```dockerfile
# Build and run
docker build -t medical-multimodal .
docker run -p 5000:5000 medical-multimodal
```

## Performance

The multimodal approach typically shows improved performance over single-modality models:
- **Text-only**: Baseline performance on clinical reports
- **Image-only**: DenseNet121 performance on X-rays
- **Multimodal**: Enhanced accuracy through information fusion

