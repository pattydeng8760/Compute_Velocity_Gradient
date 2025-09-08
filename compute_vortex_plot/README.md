# compute_vortex_plot

A modular Python package for extracting velocity invariants and generating QR plots from CFD and PIV data. This module is a refactored version of `Single_QR_Extract.py` with the same structure and logging format as `compute_vortex_detect_core`.

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for each functionality
- **CLI Interface**: Command-line interface with comprehensive argument parsing
- **Logging**: Consistent logging format matching other modules in the project
- **Data Processing**: Support for both LES and PIV data types
- **Visualization**: Comprehensive plotting capabilities for global and local invariants
- **Vortex Detection**: Multiple vortex detection methods (max, precise, area)
- **QR Analysis**: Q-R invariant extraction and plotting along PCA axes

## Module Structure

```
compute_vortex_plot/
├── __init__.py              # Module initialization and exports
├── __main__.py              # CLI entry point
├── vortex_plot.py           # Main orchestrator class
├── utils.py                 # Logging and utility functions
├── data_loader.py           # Data loading from HDF5 files
├── grid_maker.py            # Grid interpolation utilities
├── vortex_detector.py       # Vortex detection algorithms
├── invariant_extractor.py   # Velocity invariant extraction
├── single_plotter.py               # Plot generation utilities
├── data_saver.py            # Save extracted data to HDF5
└── README.md                # This file
```

## Installation

The module is part of the Compute_Velocity_Gradient project and requires the following dependencies:

- numpy
- matplotlib
- scipy
- h5py
- scikit-learn
- window_bounds (project dependency)

## Usage

### Command Line Interface

```bash
# Basic usage
python -m compute_vortex_plot --cut PIV1 --data-type LES

# With custom parameters
python -m compute_vortex_plot --cut PIV1 --data-type LES --chord 0.305 --velocity 30 --angle-of-attack 10

# Full parameter specification
python -m compute_vortex_plot \
    --cut PIV1 \
    --data-type LES \
    --chord 0.305 \
    --velocity 30 \
    --angle-of-attack 10 \
    --grid-size 500 \
    --pca-points 100 \
    --pca-length 0.012
```

### Available Arguments

- `--cut, -c`: Cutplane identifier (e.g., 'PIV1') - **Required**
- `--data-type, -d`: Data type ('LES' or 'PIV') - Default: 'LES'
- `--chord, -ch`: Chord length for normalization - Default: 0.305
- `--velocity, -v`: Free stream velocity for normalization - Default: 30
- `--angle-of-attack, -a`: Angle of attack in degrees - Default: 10
- `--grid-size, -g`: Grid size for interpolation - Default: 500
- `--pca-points, -p`: Number of PCA query points - Default: 100
- `--pca-length, -l`: PCA line length - Default: 0.012

### Python API

```python
from compute_vortex_plot import VortexPlot
import argparse

# Create arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cut', default='PIV1')
parser.add_argument('--data-type', default='LES')
# ... add other arguments
args = parser.parse_args(['--cut', 'PIV1'])

# Initialize and run
vp = VortexPlot(args)
vp.run()
```

## Workflow

The module follows this processing workflow:

1. **Data Loading**: Load velocity invariants and connectivity data from HDF5 files
2. **Grid Creation**: Create interpolation grid for global invariant plotting
3. **Vortex Detection**: Detect primary, secondary, and auxiliary vortices
4. **Invariant Extraction**: Extract velocity invariants at vortex cores and adjacent cells
5. **Plot Generation**: Generate global invariant plots and local QR analysis plots
6. **Data Saving**: Save extracted data to combined HDF5 file

## Input Requirements

The module requires the following input files:

- `Velocity_Invariants_{location}_{data_type}/Velocity_Invariants_{location}.h5`: Main velocity invariant data
- `Velocity_Invariants_{location}_{data_type}/Velocity_Invariants_{location}_Mean.h5`: Connectivity data
- `Vortex_Detect_Results_{location}_{data_type}/P_core_{location}.npy`: Primary vortex core data
- `Vortex_Detect_Results_{location}_{data_type}/S_core_{location}.npy`: Secondary vortex core data

## Output Files

The module generates the following output files:

### Global Plots
- `Global_Velocity_Invariants_{location}.png`: Global invariant distribution plots

### Local QR Analysis Plots
- `Local_Velocity_Invariants_QR_{location}_{vortex_type}.png`: Q-R invariant plots
- `Local_Velocity_Invariants_Qs_Rs_{location}_{vortex_type}.png`: Strain rate invariant plots
- `Local_Velocity_Invariants_Qs_Qw_{location}_{vortex_type}.png`: Strain-vorticity invariant plots

### PCA Profile Plots
- `Local_Velocity_Invariants_PCA_{location}_primary.png`: Primary vortex PCA profile
- `Local_Velocity_Invariants_PCA_{location}_secondary.png`: Secondary vortex PCA profile

### Data Files
- `Velocity_Invariants_Core_B_10AOA_LES_U30.h5`: Combined extracted core data (LES)
- `Velocity_Invariants_Core_B_10AOA_PIV_U30.h5`: Combined extracted core data (PIV)
- `log_vortex_plot_{location}_{data_type}.txt`: Processing log file

## Data Processing

### Supported Data Types

- **LES**: Large Eddy Simulation data from CFD simulations
  - Coordinates: X (streamwise), Y (wall normal), Z (spanwise)
  - Data path: `Velocity_Invariants_{location}_LES/`
- **PIV**: Particle Image Velocimetry experimental data
  - Original coordinates: X (wall normal), Y (spanwise), Z (streamwise)
  - Automatically mapped to LES coordinate system for processing
  - Data path: `Velocity_Invariants_{location}_PIV/`
  - Uses PIV-specific window boundaries and detection parameters

### Vortex Detection Methods

- **max**: Maximum-based detection using peak vorticity
- **precise**: Precise detection with refined criteria and airfoil masking (preferred for LES)
- **area**: Area-based detection using center of largest connected area (preferred for PIV)

### PIV-Specific Processing

The module automatically handles PIV data differences:

- **Coordinate Mapping**: PIV coordinates are mapped to LES system for consistent processing
- **Window Boundaries**: Uses PIV-specific window boundaries from `get_window_boundaries_PIV()`
- **Detection Parameters**: Adjusted vorticity thresholds and detection methods for PIV data
- **Grid Bounds**: Automatic detection of PIV data bounds for grid interpolation
- **Data Output**: Separate HDF5 files for PIV results

### Invariant Types

- **P, Q, R**: Velocity gradient tensor invariants
- **Qs, Rs**: Normalized strain rate tensor invariants
- **Qw**: Normalized vorticity tensor invariants

## Logging

The module uses the same logging format as other modules in the project:

- Console output with timestamps
- Log file: `log_vortex_plot_{location}_{data_type}.txt`
- Processing time tracking with timer decorators

## Error Handling

The module includes comprehensive error handling for:

- Missing input files
- Invalid vortex detection parameters
- Grid interpolation failures
- Data extraction errors

## Testing

Run the test suite to verify module functionality:

```bash
python test_vortex_plot.py
```

The test suite checks:
- Module imports
- CLI interface functionality
- Class initialization
- Module structure integrity

## Integration

This module integrates with the existing project workflow:

1. **First**: Run `compute_velocity_gradient_core` to compute velocity invariants
2. **Second**: Run `compute_vortex_detect_core` to detect vortex cores
3. **Third**: Run `compute_vortex_plot` to generate QR plots and extract invariants

## Dependencies on Other Modules

- `window_bounds.py`: Window boundary definitions for vortex detection
- Velocity invariant data from `compute_velocity_gradient_core`
- Vortex core data from `compute_vortex_detect_core`

## Performance Notes

- Grid interpolation is memory-intensive for large datasets
- PCA computations scale with number of vortex core points
- Plotting operations can be slow for high-resolution outputs

## Maintenance

When modifying the module:

1. Maintain consistency with the original `Single_QR_Extract.py` workflow
2. Keep the same logging format as `compute_vortex_detect_core`
3. Update tests when adding new functionality
4. Ensure backward compatibility with existing data files