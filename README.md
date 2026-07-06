# Pelagic Vision: Underwater Plastic Detection & Volume Estimation

## Project Overview
Pelagic Vision is an advanced computer vision framework designed to identify and measure the volume of plastic debris in underwater environments. Utilizing a **YOLOv8-based instance segmentation** model, this application provides real-time detection, classification, and volumetric analysis to assist in marine pollution research and environmental monitoring.

## Key Features
*   **Deep Learning Pipeline**: Built on the YOLOv8 architecture, optimized for high-speed instance segmentation of marine debris.
*   **Volumetric Analysis**: Implements an elliptical cylinder approximation algorithm to estimate the physical volume ($cm^3$) of detected plastic items based on pixel-to-centimeter calibration.
*   **Modern Web Interface**: A custom, high-performance web dashboard developed with **FastAPI** and asynchronous JavaScript, offering a sleek, dark-themed marine UI.
*   **Scalable Deployment**: Fully containerized using **Docker**, ensuring consistency across development and production environments on cloud platforms like Hugging Face Spaces.

## Architecture
*   **Model**: YOLOv8-seg (Segmented)
*   **Backend**: FastAPI
*   **Frontend**: HTML5 / CSS3 / JavaScript (Glassmorphism UI)
*   **Deployment**: Docker

## Installation & Local Development
To run this application locally, ensure you have [Docker](https://www.docker.com/) installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/underwater_plastic_detection.git](https://github.com/YOUR_USERNAME/underwater_plastic_detection.git)
    cd underwater_plastic_detection
    ```
2.  **Build the Docker image:**
    ```bash
    docker build -t pelagic-vision .
    ```
3.  **Run the container:**
    ```bash
    docker run -p 7860:7860 pelagic-vision
    ```
4.  Access the app at `http://localhost:7860`.

## Academic Context
This project was developed as part of an M.Tech/B.E. research initiative at **Chaitanya Bharathi Institute of Technology (CBIT)**. The framework explores dimensional resolution and explainability in genomic and environmental artificial intelligence.

## Supervisors & Collaborators
*   **Academic Supervision**: Dr. T. Srinivas, K. Swathi, and G. Srikanth
*   **Collaborators**: V. Lavanya
