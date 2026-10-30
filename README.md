# 🌐 RESONANCE — Federated Traffic Intelligence

### AIORI-3 Hackathon | Problem Statement PS-003

**Federated Traffic-Flow Prediction with Communication-Cost Evaluation**

<p align="center">
  <img src="https://img.shields.io/badge/Team-RESONANCE-blue" />
  <img src="https://img.shields.io/badge/Python-3.12-blue" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red" />
</p>

---

## 🚀 Project Overview

RESONANCE explores federated learning for traffic-flow prediction across a network of distributed traffic sensors.

The project compares four approaches:

- Centralized Neural Network
- Local-only Learning
- Standard Federated Averaging (FedAvg)
- Topology-aware Federated Averaging

Our objective is to investigate how different sensor-to-client assignments affect prediction accuracy, training time, and communication cost.

## 📊 Experiment at a Glance

| Parameter | Value |
|---|---:|
| Traffic sensors | 36 |
| Federated clients | 3 |
| Sensors per client | 12 |
| Communication rounds | 10 |
| Input features per sensor | 48 |
| Prediction horizon | 15 minutes |
| Prediction model | Multi-Layer Perceptron (MLP) |

## 🖥️ Interactive Dashboard

The project includes an interactive Streamlit dashboard for exploring the experiments and their results.

![Dashboard Overview](docs/images/dashboard_overview.png)

### Dashboard Features

- Project overview and experiment statistics
- Interactive sensor network visualization
- Federated training and convergence analysis
- Model performance comparison
- Communication cost analysis

## 🕸️ Sensor Network Topology

The dataset contains 36 traffic sensors connected through an adjacency matrix.

The topology-aware approach groups connected sensors into three federated clients.

- Blue: Client 1
- Yellow: Client 2
- Green: Client 3
- Grey connections: Internal client connections
- Red connections: Cross-client connections

![Sensor Network Topology](docs/images/sensor_network.png)

## 🧠 Methodology

### 1. Centralized Neural Network

A single neural network is trained using data from all 36 sensors.

### 2. Local-only Learning

Three independent models are trained locally by separate clients, without global model aggregation.

### 3. Standard FedAvg

Three clients independently train local models and send model updates to a central server. The server aggregates the updates using Federated Averaging.

Sensors are assigned in consecutive groups.

### 4. Topology-aware FedAvg

The sensor adjacency matrix is used to create connected client groups before applying Federated Averaging.

This approach investigates whether preserving sensor-network structure during client assignment affects prediction performance and communication.

**Note:** The prediction model is an MLP, not a Graph Neural Network. Network topology is used for client partitioning.

## 📈 Experimental Results

The models are evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### MAE Comparison

![MAE Comparison](docs/images/comparison_mae.png)

### RMSE Comparison

![RMSE Comparison](docs/images/comparison_rmse.png)

### Training Time Comparison

![Training Time Comparison](docs/images/training_time.png)

### Federated Training Convergence

![Federated Convergence](docs/images/convergence.png)

### Communication Cost Comparison

![Communication Cost](docs/images/communication.png)

### Final Model Comparison

| Approach | MAE | RMSE | Training Time |
|---|---:|---:|---:|
| Centralized NN | 0.028250 | 0.040177 | 21.38 s |
| Local-only | 0.026849 | 0.038428 | 15.81 s |
| Standard FedAvg | 0.028151 | 0.040374 | 13.79 s |
| Topology FedAvg | 0.033065 | 0.045063 | 63.81 s |

The Local-only approach achieved the lowest MAE and RMSE in these experiments. Topology-aware FedAvg did not achieve the best prediction accuracy, highlighting the trade-offs involved in client partitioning and federated training.

*Training times are observed execution times from these runs and are not hardware-independent benchmarks.*

## 📡 Communication Cost Analysis

Both federated experiments use the following configuration:

| Parameter | Value |
|---|---:|
| Federated clients | 3 |
| Communication rounds | 10 |
| Estimated communication per round | 176,664 bytes |
| Estimated cumulative communication | 1,766,640 bytes |

Communication is estimated from model-payload exchange. These figures exclude network protocol overhead and are not measurements from an actual distributed network deployment.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Neural network training |
| NumPy | Numerical operations |
| SciPy | MATLAB dataset loading |
| Pandas | Data handling and analysis |
| Matplotlib | Experimental graphs |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| NetworkX | Sensor network visualization |

## ▶️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/aiori-hackathon/A3-PS003-TC141.git
cd A3-PS003-TC141
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open the local address displayed in the terminal, usually:

`http://localhost:8501`

## 📁 Project Structure

```text
A3-PS003-TC141/
│
├── app.py
│
├── data/
│   └── raw/
│
├── docs/
│   └── images/
│       ├── dashboard_overview.png
│       ├── sensor_network.png
│       ├── comparison_mae.png
│       ├── comparison_rmse.png
│       ├── training_time.png
│       ├── convergence.png
│       └── communication.png
│
├── results/
│   ├── final_analysis/
│   ├── topology_fedavg/
│   └── sensor_partition.json
│
├── src/
│   ├── baselines/
│   ├── data/
│   ├── federated/
│   ├── models/
│   └── visualization/
│
├── requirements.txt
└── README.md
```

## ⚠️ Limitations

- Experiments are conducted in a simulated federated-learning environment.
- Communication cost is estimated from model-payload sizes rather than measured over a real network.
- Topology-aware partitioning uses adjacency information; the prediction model itself does not perform graph-based learning.
- Differential privacy and secure aggregation are not implemented.
- Training-time measurements depend on the execution environment.

## 👥 Team RESONANCE

**Hackathon:** AIORI-3

**Problem Statement:** PS-003 — Federated Traffic-Flow Prediction with Communication-Cost Evaluation

### 🎥 Team Introduction Video

[▶ Watch our team introduction video](https://youtu.be/FC3wcykLklc)

### 🔗 Project Repository

[GitHub — A3-PS003-TC141](https://github.com/aiori-hackathon/A3-PS003-TC141)

---

<p align="center">
  <b>AIORI-3 Hackathon | Team RESONANCE</b>
  <br/>
  Federated Traffic Intelligence
</p>