# Tennis Serve Simulator

A 3D physics-based tennis serve simulation that models realistic ball trajectories and player interactions.

## Features

- Realistic 3D ball trajectory simulation with air resistance
- Customizable serve parameters:
  - Player height (1.50m - 2.20m)
  - Serve speed (100-250 km/h)
  - Serving side (Deuce/Ad)
  - Serve direction (T, Body, Wide)
- Interactive 3D visualization
- Real-time serve statistics
- Return probability calculation based on player attributes

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- tkinter

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage

Run the simulator:
```bash
python main.py
```

## Physics Model

The simulation includes:
- Gravitational effects
- Air resistance (drag)
- Magnus effect
- Player biomechanics

## Controls

- Adjust serve parameters using sliders and dropdowns
- Click "Simulate Serve" to run simulation
- Click "Reset Simulation" to clear and start over

## Author

Martin
Physics 2260 Final Project
