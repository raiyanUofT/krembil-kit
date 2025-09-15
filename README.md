# EEG Analysis Pipeline

A comprehensive Python package for EEG signal processing, trigger detection, and frequency-domain analysis.

## Overview

This package provides a complete pipeline for analyzing EEG data stored in European Data Format (EDF) files. It includes tools for signal loading, trigger detection, inter-trigger window analysis, and multi-band frequency-domain processing.

## Features

- **EDF File Loading**: Load and inspect EEG signals with flexible duration and channel selection
- **Trigger Detection**: Automated detection of trigger events with customizable thresholds
- **Window Analysis**: Generate and analyze inter-trigger intervals with multiple aggregation methods
- **Advanced Spectral Analysis**: Multi-band power analysis with temporal smoothing and spectral parameterization
- **Connectivity Analysis**: Graph-based network analysis with correlation, coherence, and phase metrics
- **Traditional Analysis**: Multi-band EEG analysis (Delta, Theta, Alpha, Beta, Gamma) 
- **Spectral Parameterization**: Separate aperiodic (1/f) and periodic (oscillatory) components using SpecParam
- **Memory-Safe Processing**: HDF5-based analysis for large EEG files with bounded memory usage
- **Professional Organization**: Structured output directories with consistent naming conventions
- **Analysis Metadata**: Complete tracking of analysis parameters, timing, and results
- **Visualizations**: Plots and comprehensive analysis reports
- **ML Integration**: Optional machine learning-based window quality filtering

## Installation

```bash
pip install krembil-kit
```

## Quick Start

```python
from krembil_kit import EDFLoader, TriggerDetector, SpectralAnalyzer, ConnectivityAnalyzer

# Load EEG data
loader = EDFLoader("data", "subject_name")
loader.load_and_plot_signals(signal_indices=[15, 25], duration=1200.0)  # T6, T2

# Detect and plot triggers for temporal segmentation
detector = TriggerDetector(loader, 'T2')
detector.detect_triggers()
detector.plot_triggers()

# Option 1: Advanced spectral analysis
spectral_analyzer = SpectralAnalyzer(loader=loader, trigger_detector=detector)
spectral_analyzer.analyze_comprehensive()  # Multi-band + spectral parameterization

# Option 2: Graph-based connectivity analysis
connectivity_analyzer = ConnectivityAnalyzer(edf_loader=loader)

# Level 1: Quick exploration
connectivity_analyzer.compute_correlation(start_time=0, stop_time=300, interval=10)
connectivity_analyzer.compute_coherence_average(start_time=0, stop_time=300, interval=30)

# Level 2: Detailed analysis  
connectivity_analyzer.compute_coherence_bands(start_time=0, stop_time=300, interval=50)

# Plot results
connectivity_analyzer.plot_connectivity_matrices()

# Level 3: Full graph representations
# Memory-safe graph generation (works for any file size):
hdf5_path = connectivity_analyzer.generate_graphs(segment_duration=60.0, overlap_ratio=0.125)
```

## Data Structure Requirements

### Input Data Format

Your EDF files should be organized as follows:

```
data/
└── subject_name/
    └── subject_name.edf
```

### EDF File Requirements

- **Format**: European Data Format (.edf)
- **Channels**: Standard EEG channel names (Fp1, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz)
- **Sample Rate**: Typically 500 Hz (automatically detected)
- **Duration**: Minimum 10 minutes recommended for trigger detection

## Classes and Methods

### EDFLoader

Handles loading and inspection of EDF files.

#### Initialization
```python
loader = EDFLoader(folder_path, name)
```

**Parameters:**
- `folder_path` (str): Path to the data directory
- `name` (str): Name of the subject or experiment (the EDF file will be under this subject directory, and should match this name)

For example, if our data was in the current directory as below:

```
data/
└── Sebastian/
    └── Sebastian.edf
```

We would call (from the current directory):

```python
loader = EDFLoader("path/to/data/folder", "Sebastian")
```

#### Methods

##### `inspect_data()`
Inspects the EDF file for the specified subject, printing out various signal information including:
- File header details
- Number of signals and their properties
- Sample rates and signal ranges
- Physical and digital maximum/minimum values
- First 10 samples of each channel

```python
loader.inspect_data()
```

##### `load_and_plot_signals(signal_indices=None, duration=None, save_plots=False, save_path=None)`
Loads and plots the signals from the EDF file, storing them along with their sample rates in a dictionary.

**Parameters:**
- `signal_indices` (list of int, optional): Specific signal indices to load (None = load all signals)
- `duration` (float, optional): Duration in seconds to plot (None = entire duration)
- `save_plots` (bool, optional): Whether to save plots instead of showing them (default: False)
- `save_path` (str, optional): Directory to save plots (default: plots/{subject_name})

**Examples:**
```python
# Load T6 and T2 channels for 20 minutes
loader.load_and_plot_signals(signal_indices=[15, 25], duration=1200.0)

# Load all channels and save plots
loader.load_and_plot_signals(save_plots=True)

# Load specific duration with custom save path
loader.load_and_plot_signals(duration=1200.0, save_plots=True, save_path="custom_plots")
```

**Output:**
- Time-series plots with time axis in seconds
- Signals stored in `signals_dict` attribute with data and sample rates
- Plots saved to `plots/{subject_name}/signals_plot.png` (if save_plots=True)
- Warning message displayed when loading all signals due to potential memory usage

### TriggerDetector

Detects triggers and analyzes inter-trigger windows using amplitude thresholding and machine learning-based quality filtering.

#### Initialization
```python
detector = TriggerDetector(edf_loader, signal_choice)
```

**Parameters:**
- `edf_loader` (EDFLoader): An instance of the EDFLoader class that contains the signal data
- `signal_choice` (str): The key corresponding to the desired signal in the EDFLoader's signals_dict (e.g., 'T2', 'O1')

#### Methods

##### `butterworth_filter(signal, cutoff=30, order=5, btype='low')`
Applies a Butterworth filter to the signal.

**Parameters:**
- `signal` (numpy.ndarray): The input signal data to filter
- `cutoff` (float): The cutoff frequency in Hz (default: 30)
- `order` (int): The order of the filter (default: 5)
- `btype` (str): The type of the filter ('low', 'high', 'bandpass', 'bandstop') (default: 'low')

**Returns:** numpy.ndarray - The filtered signal

##### `detect_triggers()`
Detects triggers in the signal based on a threshold and filters events to be between 55 seconds and 62 seconds long.

**Algorithm:**
1. Rectifies the signal using absolute value
2. Applies Butterworth low-pass filter (30 Hz cutoff, order 5)
3. Detects events above hardcoded threshold (60 µV)
4. Filters events by duration (52-65 seconds, corresponding to 55-62 second range)
5. Handles edge cases for events at signal boundaries

```python
detector.detect_triggers()
print(f"Found {len(detector.df_triggers)} triggers")
```

**Output:**
- `df_triggers` DataFrame with columns:
  - `start_index`, `end_index`: Sample indices
  - `duration_samples`: Duration in samples
  - `start_time (s)`, `end_time (s)`: Time in seconds
  - `duration_time (s)`: Trigger duration in seconds

##### `plot_triggers()`
Plots the filtered signal with detected trigger periods highlighted.

```python
detector.plot_triggers()
```

**Output:** Interactive matplotlib plot showing filtered signal with red highlighted trigger periods

##### `save_triggers()`
Saves the DataFrame of detected triggers to a CSV file.

```python
detector.save_triggers()
```

**Output:** `{subject_folder}/triggers.csv`

##### `plot_windows()`
Generates individual plots for each inter-trigger window (signal segments between consecutive triggers).

```python
detector.plot_windows()
```

**Output:** `{subject_folder}/window plots/plot_{i}.png` - Individual plots for each window with time in minutes and amplitude range 0-300

##### `convert_to_video()`
Creates MP4 video from window plots for rapid review, sorted numerically by plot index.

```python
detector.convert_to_video()
```

**Output:** `{subject_folder}/trigger.mp4` - Video at 10 fps showing all window plots in sequence

##### `filter_bad_windows(clf_path=None, classes_path=None)`
Uses ResNet-50 + logistic regression pipeline to drop triggers whose adjoining window plots are classified as 'bad'. Automatically uses built-in models when no custom paths are provided.

**Important:** Call after `plot_windows()` to ensure window plots exist for classification.

```python
# Use built-in models (recommended)
detector.plot_windows()
detector.filter_bad_windows()

# Or use custom models
detector.filter_bad_windows(
    clf_path="path/to/custom_classifier.pkl",
    classes_path="path/to/custom_classes.npy"
)
```

**Parameters:**
- `clf_path` (str, optional): Path to custom classifier (.pkl file). Uses built-in model from package resources if None
- `classes_path` (str, optional): Path to custom class labels (.npy file). Uses built-in classes from package resources if None

**Output:** Overwrites `{subject_folder}/triggers.csv` with filtered results, removing triggers adjacent to windows classified as 'bad'





### SpectralAnalyzer

Advanced spectral analysis tool providing comprehensive frequency-domain characterization of EEG signals with both time-domain power analysis and modern spectral parameterization methods. Features professional output organization and complete analysis metadata tracking.

The SpectralAnalyzer offers **two complementary analysis approaches**:

1. **Multi-band Power Analysis** - Time-resolved power across canonical EEG frequency bands
2. **Spectral Parameterization** - SpecParam analysis separating aperiodic and periodic components

#### Initialization
```python
from krembil_kit import EDFLoader, TriggerDetector, SpectralAnalyzer

# Load EEG data and detect triggers
loader = EDFLoader(folder_path="data", name="subject_name")
loader.load_and_plot_signals()

trigger_detector = TriggerDetector(edf_loader=loader, signal_choice='T2')
trigger_detector.detect_triggers()

# Initialize analyzer with optional custom output directory
analyzer = SpectralAnalyzer(
    loader=loader, 
    trigger_detector=trigger_detector, 
    target_length=50,
    output_dir=None  # Optional: defaults to subject_folder/spectral_analysis_results/
)
```

**Parameters:**
- `loader` (EDFLoader): Configured EDFLoader instance containing loaded EEG signals. Must have signals loaded via load_and_plot_signals() method.
- `trigger_detector` (TriggerDetector, optional): TriggerDetector instance for event-based signal segmentation. Required for multi-band power analysis with temporal segmentation.
- `target_length` (int, default=50): Number of resampled points per segment for temporal aggregation. Controls the resolution of time-series outputs.
- `output_dir` (str, optional): Output directory path for analysis results. If None, defaults to 'spectral_analysis_results' subdirectory in the same directory as the EDF file.

#### Methods

##### `analyze_multiband_power(channels_to_analyze=None)`

Executes comprehensive multi-band power analysis across canonical EEG frequency bands (Delta, Theta, Alpha, Beta, Gamma) with configurable temporal smoothing.

**Features:**
- Butterworth bandpass filtering for frequency band isolation
- Signal rectification and moving-average smoothing
- Multiple smoothing windows (100ms, 250ms, 500ms) for different temporal scales
- Structured CSV output and publication-ready visualizations

```python
# Analyze all loaded channels
analyzer.analyze_multiband_power()

# Analyze specific channels
analyzer.analyze_multiband_power(channels_to_analyze=['T2', 'O1', 'F3'])
```

**Output Structure:**
```
subject_folder/
└── spectral_analysis_results/
    ├── multiband_power/
    │   ├── csv/
    │   │   ├── subject_multiband_Delta_ma100ms.csv
    │   │   ├── subject_multiband_Theta_ma250ms.csv
    │   │   └── subject_multiband_Gamma_ma500ms.csv
    │   └── plots/
    │       ├── subject_multiband_Delta_T2.png
    │       └── subject_multiband_Theta_T2.png
    └── analysis_metadata.json  # Complete analysis tracking
```

##### `analyze_spectral_parameterization(channels_to_analyze=None)`

Executes advanced spectral parameterization using SpecParam methodology to separate neural power spectra into aperiodic (1/f) and periodic (oscillatory) components.

**Features:**
- Modern SpecParam library for spectral parameterization
- Robust model fitting with configurable parameters
- Comprehensive validation metrics and goodness-of-fit assessment
- Frequency band power quantification with aperiodic correction

```python
# Analyze all loaded channels
analyzer.analyze_spectral_parameterization()

# Analyze specific channels
analyzer.analyze_spectral_parameterization(channels_to_analyze=['T2', 'O1'])
```

**Output Structure:**
```
subject_folder/
└── spectral_analysis_results/
    ├── spectral_parameterization/
    │   ├── individual/
    │   │   ├── subject_fooof_T2.png
    │   │   ├── subject_fooof_parameters_T2.csv
    │   │   └── subject_band_powers_T2.csv
    │   ├── summary/
    │   │   ├── subject_fooof_parameters_summary.csv
    │   │   └── subject_band_powers_summary.csv
    │   └── plots/
    │       ├── subject_aperiodic_exponent_comparison.png
    │       └── subject_spectral_peaks_comparison.png
    └── analysis_metadata.json  # Complete analysis tracking
```

##### `analyze_comprehensive(channels_to_analyze=None)`

Executes complete spectral analysis suite combining both multi-band power analysis and spectral parameterization for comprehensive frequency-domain characterization.

```python
# Complete analysis workflow
analyzer.analyze_comprehensive(channels_to_analyze=['T2', 'O1', 'F3'])
```

#### Configuration Methods

##### `set_frequency_bands(bands_dict)`

Configure custom frequency bands for multi-band analysis.

```python
# Custom frequency bands
custom_bands = {
    'slow_alpha': (8, 10),
    'fast_alpha': (10, 12),
    'low_beta': (12, 20),
    'high_beta': (20, 30)
}
analyzer.set_frequency_bands(custom_bands)
```

##### `set_fooof_parameters(freq_range=None, **fooof_kwargs)`

Configure spectral parameterization parameters.

```python
# Custom SpecParam settings
analyzer.set_fooof_parameters(
    freq_range=(1, 40),
    peak_width_limits=(1, 8),
    max_n_peaks=6,
    min_peak_height=0.1
)
```

##### `set_smoothing_windows(window_secs_list)`

Configure temporal smoothing parameters for multi-band power analysis.

**Parameters:**
- `window_secs_list` (list of float): Moving-average window sizes in seconds. Multiple windows enable comparison of different temporal smoothing scales.

**Notes:**
- Shorter windows preserve temporal dynamics but may be noisier
- Longer windows provide smoother estimates but reduce temporal resolution
- Window sizes are automatically converted to samples based on signal sampling frequency during analysis

```python
# Custom smoothing windows
analyzer.set_smoothing_windows([0.05, 0.1, 0.25])  # 50ms, 100ms, 250ms
```

##### `get_analysis_info()`

Retrieve comprehensive analysis configuration and system information for documentation and reproducibility purposes.

**Returns:**
- `dict`: Configuration dictionary containing channels, frequency bands, smoothing windows, spectral parameterization settings, library information, and trigger detector status.

```python
# Get current analyzer configuration
config = analyzer.get_analysis_info()
print(f"Available channels: {config['channels']}")
print(f"Frequency bands: {config['frequency_bands']}")
print(f"Library: {config['spectral_param_library']['library']}")
```

#### Visualization Methods

##### `plot_raw_signal_window(window_index, channel)`

Generate publication-ready visualization of raw EEG data for specified trigger-defined window.

**Parameters:**
- `window_index` (int): Zero-based index of the trigger-defined window to visualize. Must be less than (total_triggers - 1) to ensure valid window bounds.
- `channel` (str): EEG channel name to plot. Must exist in loaded dataset.

**Raises:**
- `ValueError`: If trigger detector is not provided or window_index is out of range.

```python
# Plot specific window
analyzer.plot_raw_signal_window(window_index=5, channel='T2')
```

##### `plot_averaged_signal_window(channel, start_window=None, end_window=None, target_length=500, aggregation_method='mean', trim_ratio=0.1)`

Create ensemble-averaged signal visualization across multiple temporal windows with robust statistical aggregation.

**Parameters:**
- `channel` (str): EEG channel name to analyze and visualize
- `start_window` (int, optional): Starting window index for aggregation. If None, uses first window (0)
- `end_window` (int, optional): Ending window index for aggregation. If None, uses last available window
- `target_length` (int, default=500): Number of temporal points for signal resampling and standardization
- `aggregation_method` ({'mean', 'median', 'trimmed'}, default='mean'): Statistical method for cross-window aggregation
- `trim_ratio` (float, default=0.1): Proportion of extreme values to exclude for 'trimmed' aggregation method

```python
# Plot averaged signal across windows 10-20
analyzer.plot_averaged_signal_window(
    channel='T2',
    start_window=10,
    end_window=20,
    aggregation_method='median'
)

# Plot with trimmed mean aggregation
analyzer.plot_averaged_signal_window(
    channel='T2',
    aggregation_method='trimmed',
    trim_ratio=0.2
)
```

##### `plot_fooof_comparison(channels=None, metric='aperiodic_exponent')`

Generate comparative visualization of spectral parameterization metrics across channels.

**Parameters:**
- `channels` (list of str, optional): EEG channel names to include in comparison. If None, includes all channels with completed spectral parameterization analysis.
- `metric` ({'aperiodic_exponent', 'aperiodic_offset', 'n_peaks', 'r_squared', 'error'}, default='aperiodic_exponent'): Spectral parameterization metric to visualize:
  - 'aperiodic_exponent': 1/f slope reflecting neural population dynamics
  - 'aperiodic_offset': Broadband power offset parameter
  - 'n_peaks': Number of detected oscillatory peaks
  - 'r_squared': Model fit quality (coefficient of determination)
  - 'error': Root mean square error of model fit

**Notes:**
- Requires prior execution of analyze_spectral_parameterization() method
- Visualization includes professional formatting with grid lines, proper axis labels, and publication-ready styling

```python
# Compare aperiodic exponents across channels
analyzer.plot_fooof_comparison(
    channels=['T2', 'O1', 'F3'],
    metric='aperiodic_exponent'
)

# Compare number of peaks across all analyzed channels
analyzer.plot_fooof_comparison(metric='n_peaks')
```

#### Complete Analysis Example

```python
from krembil_kit import EDFLoader, TriggerDetector, SpectralAnalyzer

# Step 1: Load data and detect triggers
loader = EDFLoader(folder_path="data", name="subject_name")
loader.load_and_plot_signals()

trigger_detector = TriggerDetector(edf_loader=loader, signal_choice='T2')
trigger_detector.detect_triggers()
trigger_detector.plot_triggers()

# Step 2: Initialize analyzer
analyzer = SpectralAnalyzer(
    loader=loader,
    trigger_detector=trigger_detector
)

# Step 3: Configure analysis parameters
analyzer.set_frequency_bands({
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 80)
})

analyzer.set_fooof_parameters(freq_range=(1, 40))

# Step 4: Execute comprehensive analysis
channels_of_interest = ['T2', 'O1', 'F3', 'C3', 'C4']
analyzer.analyze_comprehensive(channels_to_analyze=channels_of_interest)

# Step 5: Generate comparative visualizations
analyzer.plot_fooof_comparison(
    channels=channels_of_interest,
    metric='aperiodic_exponent'
)
```

### ConnectivityAnalyzer

Converts EEG data to graph representations for network analysis and computes time-varying connectivity measures. Features professional output organization, complete analysis metadata tracking, and memory-efficient processing for any file size.

The ConnectivityAnalyzer provides **three levels of analysis complexity** to match different research needs:

1. **Simple Connectivity Analysis** - Fast correlation/coherence for exploration
2. **Detailed Connectivity Analysis** - Time-varying connectivity with visualizations  
3. **Advanced Graph Analysis** - Full graph representations for machine learning

#### Initialization
```python
processor = ConnectivityAnalyzer(
    edf_loader=loader,
    output_dir=None,            # Optional: custom output directory
    window_size=1000,           # Optional: analysis window size in samples
    adj_window_size=20000       # Optional: adjacency matrix window size (40s at 500Hz)
)
```

**Key Parameters:**
- **`window_size`**: Analysis window duration (default: 1 second at sampling rate)
- **`adj_window_size`**: Window size for adjacency calculations (default: 40 seconds for statistical robustness)
- **`output_dir`**: Custom output directory (default: subject_folder/connectivity_analysis_results/)

**Default Output Structure:**
```
data/subject_name/
├── subject_name.edf
└── connectivity_analysis_results/  # Professional organization
    ├── graphs/                     # HDF5 graph representations
    │   └── subject_graphs.h5
    ├── correlation/                # Correlation matrices
    │   ├── subject_correlation_0s-300s.pickle
    │   └── plots/                  # Correlation visualizations
    ├── coherence/
    │   ├── average/                # Average coherence matrices
    │   │   ├── subject_coherence_avg_0s-300s.pickle
    │   │   └── plots/              # Average coherence visualizations
    │   └── bands/                  # Frequency-band coherence matrices
    │       ├── subject_coherence_bands_0s-300s.pickle
    │       └── plots/              # Band-specific visualizations
    └── analysis_metadata.json     # Complete analysis tracking
```

#### Methods

##### `generate_graphs(segment_duration=180.0, start_time=None, stop_time=None, overlap_ratio=0.875)`
Creates comprehensive graph representations with adjacency matrices and node/edge features using **memory-safe HDF5 format** with segmented processing.

**Features Generated:**
- **Adjacency matrices**: Correlation, coherence, phase relationships
- **Node features**: Energy, band-specific energy across frequency bands
- **Edge features**: Connectivity measures across frequency bands
- **High temporal resolution**: 87.5% overlapping windows by default

**Key Advantages:**
- **Memory-safe**: Processes any file size without memory issues
- **Segmented processing**: Divides large files into manageable segments
- **Immediate storage**: Results saved incrementally to prevent data loss
- **Progress tracking**: Real-time progress bars and detailed logging
- **HDF5 format**: Compressed, efficient storage with selective data access

**Parameters:**
- `segment_duration` (float): Duration of each processing segment in seconds (default: 180.0)
- `start_time` (float, optional): Start time for analysis window in seconds
- `stop_time` (float, optional): End time for analysis window in seconds  
- `overlap_ratio` (float): Window overlap ratio (default: 0.875 = 87.5% overlap)

```python
# Generate comprehensive graph representations
hdf5_path = processor.generate_graphs(segment_duration=300.0)

# Analyze specific time window with high temporal resolution
hdf5_path = processor.generate_graphs(
    segment_duration=180.0,
    start_time=300,
    stop_time=900,
    overlap_ratio=0.95  # Very high resolution
)
# Output: graphs/{filename}_graphs.h5 with compressed graph data
```

**HDF5 Output Structure:**
```python
# HDF5 file contains:
{
    'adjacency_matrices': (n_windows, n_adj_types, n_electrodes, n_electrodes),
    'node_features': (n_windows,),  # Variable-length arrays
    'edge_features': (n_windows,),  # Variable-length arrays  
    'window_starts': (n_windows,),  # Timestamp for each window
    # Plus comprehensive metadata as attributes
}
```

**Loading HDF5 Results:**
```python
import h5py
import numpy as np

# Load specific data without loading entire file
with h5py.File('subject_graphs.h5', 'r') as f:
    # Load specific time range
    correlation_matrices = f['adjacency_matrices'][100:200, 1, :, :]  # Windows 100-200, correlation type
    
    # Load metadata
    sampling_freq = f.attrs['sampling_frequency']
    total_windows = f.attrs['total_windows_processed']
    
    # Load specific electrode pairs
    electrode_pair_data = f['adjacency_matrices'][:, 1, 5, 12]  # All windows, electrodes 5-12
```

##### `compute_correlation(start_time, stop_time, interval, overlap_ratio=0.0)`
Computes time-varying correlation matrices over specified time segments.

**Parameters:**
- `start_time` (float): Start time in seconds
- `stop_time` (float): End time in seconds  
- `interval` (float): Window duration for each correlation matrix in seconds
- `overlap_ratio` (float): Overlap between windows (0.0 = no overlap, 0.5 = 50% overlap)

```python
# Compute correlation every 5 seconds from 10-60s with 50% overlap
path = processor.compute_correlation(
    start_time=10.0,
    stop_time=60.0, 
    interval=5.0,
    overlap_ratio=0.5
)
```

**Output:** `correlation/{filename}_correlation_{start}s-{stop}s.pickle` containing:
```python
{
    "starts": [10.0, 12.5, 15.0, ...],  # Window start times
    "corr_matrices": [matrix1, matrix2, ...]  # Correlation matrices
}
```

##### `compute_coherence_average(start_time, stop_time, interval, overlap_ratio=0.0)`
Computes time-varying coherence matrices averaged across all frequency bands.

```python
# Simple averaged coherence analysis
path = processor.compute_coherence_average(
    start_time=10.0,
    stop_time=60.0,
    interval=5.0
)
```

**Output:** `coherence/average/{filename}_coherence_avg_{start}s-{stop}s.pickle` containing:
```python
{
    "starts": [10.0, 15.0, 20.0, ...],
    "coherence_matrices": [matrix1, matrix2, ...]  # Averaged coherence
}
```

##### `compute_coherence_bands(start_time, stop_time, interval, overlap_ratio=0.0)`
Computes detailed frequency-specific coherence analysis across EEG bands.

```python
# Detailed frequency-band coherence analysis  
path = processor.compute_coherence_bands(
    start_time=10.0,
    stop_time=60.0,
    interval=5.0,
    overlap_ratio=0.25
)
```

**Output:** `coherence/bands/{filename}_coherence_bands_{start}s-{stop}s.pickle` containing:
```python
{
    "starts": [10.0, 15.0, 20.0, ...],
    "coherence_by_band": {
        "delta": [matrix1, matrix2, ...],    # 1-4 Hz
        "theta": [matrix1, matrix2, ...],    # 4-8 Hz  
        "alpha": [matrix1, matrix2, ...],    # 8-13 Hz
        "beta": [matrix1, matrix2, ...],     # 13-30 Hz
        "gamma": [matrix1, matrix2, ...],    # 30-70 Hz
        "gammaHi": [matrix1, matrix2, ...],  # 70-100 Hz
        # Additional bands based on sampling frequency
    },
    "frequency_bands": {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), ...
    }
}
```

##### `plot_connectivity_matrices(plot_types=None, time_range=None, output_subdir="plots", save_individual=True, save_summary=True, dpi=150, figsize=(10, 8))`
Generates comprehensive visualizations of connectivity matrices with **full EEG channel names** on axes.

**Parameters:**
- `plot_types` (list): Types to plot - `["correlation", "coherence_avg", "coherence_bands"]` (default: all available)
- `time_range` (tuple): `(start_time, stop_time)` to filter plots (default: all time windows)
- `output_subdir` (str): Subdirectory name for plots (default: "plots")
- `save_individual` (bool): Save individual matrix plots (default: True)
- `save_summary` (bool): Save summary/comparison plots (default: True)
- `dpi` (int): Plot resolution (default: 150)
- `figsize` (tuple): Figure size as (width, height) (default: (10, 8))

**Features:**
- **Full channel names**: All EEG channel names (Fp1, F3, C3, etc.) displayed on both axes
- **Organized output**: Plots saved alongside data in intuitive directory structure
- **Multiple plot types**: Individual matrices, time series summaries, frequency band comparisons
- **Flexible filtering**: Plot specific time ranges or connectivity types
- **High-quality output**: Publication-ready plots with proper labeling

```python
# Plot all available connectivity data
results = processor.plot_connectivity_matrices()

# Plot only correlation matrices
results = processor.plot_connectivity_matrices(plot_types=["correlation"])

# Plot coherence with time filtering
results = processor.plot_connectivity_matrices(
    plot_types=["coherence_avg", "coherence_bands"],
    time_range=(100, 200),  # Only plot 100-200 second window
    save_individual=True,
    save_summary=True
)

# Custom plot settings
results = processor.plot_connectivity_matrices(
    dpi=300,  # High resolution
    figsize=(12, 10),  # Larger plots
    output_subdir="publication_plots"
)
```

#### Progressive Analysis Workflow

The ConnectivityAnalyzer supports a **progressive complexity approach** - start simple and add detail as needed:

##### Level 1: Exploratory Analysis (Fast)
```python
from krembil_kit import EDFLoader, ConnectivityAnalyzer

# Load EEG data
loader = EDFLoader("data", "subject_name")
loader.load_and_plot_signals(duration=1200.0)

# Initialize processor
processor = ConnectivityAnalyzer(edf_loader=loader)

# Quick correlation overview (5-minute windows)
corr_path = processor.compute_correlation(
    start_time=0, stop_time=3600, interval=300
)

# Quick coherence overview
coh_path = processor.compute_coherence_average(
    start_time=0, stop_time=3600, interval=300
)

# Generate overview plots
processor.plot_connectivity_matrices(
    plot_types=["correlation", "coherence_avg"],
    save_individual=False,  # Only summary plots
    save_summary=True
)
```

##### Level 2: Detailed Time-Varying Analysis
```python
# Identify interesting periods from Level 1 results
# Focus on specific time ranges with higher resolution

# High-resolution analysis of interesting periods
processor_detailed = ConnectivityAnalyzer(edf_loader=loader)

# Detailed correlation analysis (10-second windows)
detailed_corr = processor_detailed.compute_correlation(
    start_time=100, stop_time=400, interval=10, overlap_ratio=0.5
)

# Frequency-specific coherence analysis
detailed_coh = processor_detailed.compute_coherence_bands(
    start_time=100, stop_time=400, interval=10, overlap_ratio=0.5
)

# Generate detailed visualizations
processor_detailed.plot_connectivity_matrices(
    plot_types=["correlation", "coherence_bands"],
    time_range=(100, 400),
    save_individual=True,
    save_summary=True
)
```

##### Level 3: Advanced Graph Analysis
```python
# For machine learning, GNN analysis, or comprehensive connectivity studies

# Memory-safe graph generation for any file size
processor_advanced = ConnectivityAnalyzer(
    edf_loader=loader
)
hdf5_path = processor_advanced.generate_graphs(
    segment_duration=180.0
)

# Load and analyze HDF5 results
import h5py
with h5py.File(hdf5_path, 'r') as f:
    # Access specific connectivity types
    correlations = f['adjacency_matrices'][:, 1, :, :]  # All correlation matrices
    coherences = f['adjacency_matrices'][:, 2, :, :]    # All coherence matrices
    
    # Get metadata
    n_windows = f.attrs['total_windows_processed']
    sampling_freq = f.attrs['sampling_frequency']
    
    print(f"Processed {n_windows} windows at {sampling_freq} Hz")
```

#### Method Selection Guide

**Use `compute_correlation()` when:**
- ✅ Quick data exploration and quality assessment
- ✅ Identifying periods of high/low connectivity
- ✅ Simple statistical comparisons between conditions
- ✅ Real-time or streaming analysis needs
- ✅ Memory-constrained environments

**Use `compute_coherence_average()` when:**
- ✅ Frequency-domain connectivity without band-specific details
- ✅ Robust connectivity measures (coherence is less sensitive to artifacts)
- ✅ Comparing connectivity strength across different time periods
- ✅ Preprocessing for more detailed analysis

**Use `compute_coherence_bands()` when:**
- ✅ Need frequency-specific connectivity (alpha, beta, gamma, etc.)
- ✅ Studying oscillatory coupling between brain regions
- ✅ Clinical applications requiring band-specific analysis
- ✅ Research into frequency-specific network dynamics

**Use `generate_graphs()` when:**
- ✅ Machine learning applications (GNNs, classification)
- ✅ Complex network analysis requiring multiple connectivity measures
- ✅ Research requiring high temporal resolution connectivity tracking
- ✅ Any size EDF files (memory-safe processing)
- ✅ Production environments requiring reliability
- ✅ Need for incremental processing and progress tracking
- ✅ Long-term storage with efficient HDF5 compression

#### Complete Analysis Example
```python
from krembil_kit import EDFLoader, ConnectivityAnalyzer
import h5py
import numpy as np

# Load EEG data
loader = EDFLoader("data", "subject_name")
loader.load_and_plot_signals(duration=3600.0)  # 1 hour

# Step 1: Quick exploration (Level 1)
explorer = ConnectivityAnalyzer(edf_loader=loader)

# Overview analysis
corr_overview = explorer.compute_correlation(0, 3600, 300)  # 5-min windows
coh_overview = explorer.compute_coherence_average(0, 3600, 300)

# Generate overview plots
explorer.plot_connectivity_matrices(
    plot_types=["correlation", "coherence_avg"],
    save_summary=True
)

# Step 2: Identify interesting periods (hypothetical analysis)
# ... analyze overview results to find periods of interest ...
interesting_start, interesting_stop = 1200, 1800  # Example: 20-30 minutes

# Step 3: Detailed analysis of interesting period (Level 2)
detailed = ConnectivityAnalyzer(edf_loader=loader)

detailed_corr = detailed.compute_correlation(
    interesting_start, interesting_stop, 30, overlap_ratio=0.5
)
detailed_coh = detailed.compute_coherence_bands(
    interesting_start, interesting_stop, 30, overlap_ratio=0.5
)

# Step 4: Full graph analysis for ML (Level 3)
# Memory-safe HDF5 processing
advanced = ConnectivityAnalyzer(edf_loader=loader)
hdf5_path = advanced.generate_graphs(segment_duration=300.0)

# Step 5: Load and analyze results
with h5py.File(hdf5_path, 'r') as f:
    # Extract features for machine learning
    correlation_features = f['adjacency_matrices'][:, 1, :, :].flatten()
    coherence_features = f['adjacency_matrices'][:, 2, :, :].flatten()
    
    # Get temporal information
    window_times = f['window_starts'][:]
    
    # Print summary
    print(f"Generated {len(window_times)} windows")
    print(f"Time range: {window_times[0]:.1f}s - {window_times[-1]:.1f}s")
    print(f"Feature dimensions: {correlation_features.shape}")

# Step 6: Generate comprehensive visualizations
advanced.plot_connectivity_matrices(
    plot_types=["correlation", "coherence_avg", "coherence_bands"],
    time_range=(interesting_start, interesting_stop),
    save_individual=True,
    save_summary=True,
    dpi=300  # High resolution for publication
)
```

## Advanced Usage

### Memory Management for Large Files

**For EDF Loading:**
- Use `duration` parameter to limit data loading
- Use `signal_indices` to select specific channels
- Enable `save_plots=True` to avoid memory issues with display

**For Graph Processing:**
- **Any file size**: `generate_graphs()` uses memory-safe HDF5 processing
- **Adjust segment size**: Smaller segments use less memory but have more boundary losses

```python
# Memory-efficient settings for large files
processor = ConnectivityAnalyzer(edf_loader=loader)

# Process in small segments for maximum memory efficiency
hdf5_path = processor.generate_graphs(
    segment_duration=120.0  # Smaller segments = less memory
)
```

### Custom Trigger Detection Parameters

The trigger detection uses hardcoded parameters optimized for trigger detection:
- **Threshold**: 60 µV
- **Duration range**: 52-65 seconds
- **Filter**: 30 Hz low-pass Butterworth

### Temporal Resolution vs Performance Trade-offs

**Overlap Ratio Impact:**
```python
# High resolution, high computational cost
hdf5_path = processor.generate_graphs(overlap_ratio=0.875)  # 87.5% overlap
# Result: 8x more windows, 8x longer processing, 8x more storage

# Moderate resolution, balanced performance  
hdf5_path = processor.generate_graphs(overlap_ratio=0.5)   # 50% overlap
# Result: 2x more windows, 2x longer processing

# Low resolution, fast processing
hdf5_path = processor.generate_graphs(overlap_ratio=0.0)   # No overlap
# Result: Fastest processing, lowest memory usage
```

### HDF5 Data Access Patterns

**Efficient HDF5 Loading:**
```python
import h5py

# Load specific time ranges without loading entire file
with h5py.File('subject_graphs.h5', 'r') as f:
    # Load only correlation matrices for specific time window
    correlations = f['adjacency_matrices'][100:200, 1, :, :]
    
    # Load specific electrode pairs across all time
    electrode_pair = f['adjacency_matrices'][:, 1, 5, 12]
    
    # Load metadata without loading data
    total_windows = f.attrs['total_windows_processed']
    sampling_freq = f.attrs['sampling_frequency']
```

### ML-Based Quality Control

For automated window quality assessment:
1. Train a ResNet-50 + logistic regression model on labeled window images
2. Save the classifier and class labels
3. Use `filter_bad_windows()` to automatically remove poor-quality segments

### Production Deployment Considerations

**For Large-Scale Processing:**
- Use `generate_graphs()` for reliability and memory safety with HDF5 storage
- Set appropriate `segment_duration` based on available RAM
- Monitor disk space - HDF5 files can be large but are compressed
- Use progress tracking to monitor long-running jobs
- Consider processing multiple files in parallel with separate processes

**Error Recovery:**
- HDF5 processing saves incrementally - partial results preserved on interruption
- Check for existing HDF5 files before reprocessing
- Use validation scripts to verify data integrity

## Dependencies

- numpy
- scipy  
- mne
- pyedflib
- matplotlib
- seaborn
- pandas
- opencv-python
- torch
- torchvision
- joblib
- scikit-learn
- Pillow
- specparam
- specparam
- h5py
- tqdm

## Requirements

- Python ≥ 3.7
- Sufficient RAM for EEG data (recommend 8GB+ for large files)
- GPU optional (for ML-based filtering)

## Citation

If you use this package in your research, please cite:

```
[KrembilKit - Raiyan, Yousif, Srikar]
```

## License

MIT License

## Support

For questions or issues, please contact the package maintainer.
#
# Analysis Metadata and Reproducibility

Both SpectralAnalyzer and ConnectivityAnalyzer automatically track comprehensive metadata for all analyses.

### Metadata Features

**Automatic Tracking:**
- Analysis timestamps and duration
- All analysis parameters and settings
- Input data information (file paths, channels, etc.)
- Results summary (files created, processing statistics)
- Software version and library information

**Metadata File Location:**
```
subject_folder/
├── spectral_analysis_results/
│   └── analysis_metadata.json
└── connectivity_analysis_results/
    └── analysis_metadata.json
```

### Metadata Structure

```json
[
  {
    "timestamp": "2024-12-19T14:30:52.123456",
    "analysis_type": "comprehensive",
    "analysis_duration_seconds": 245.7,
    "parameters": {
      "channels_analyzed": ["T2", "O1", "F3"],
      "frequency_bands": {"Delta": [0.5, 4], "Theta": [4, 8]},
      "fooof_settings": {"max_n_peaks": 6, "peak_threshold": 2.0}
    },
    "data_info": {
      "subject_name": "subject_001",
      "channels": ["T2", "O1", "F3", "C3", "C4"]
    },
    "results": {
      "analysis_type": "comprehensive",
      "methods_executed": ["multiband_power", "spectral_parameterization"],
      "channels_processed": 3
    }
  }
]
```

### Using Metadata

```python
import json

# Load analysis history
with open('spectral_analysis_results/analysis_metadata.json', 'r') as f:
    metadata = json.load(f)

# Find specific analysis
for analysis in metadata:
    if analysis['analysis_type'] == 'comprehensive':
        print(f"Analysis run on: {analysis['timestamp']}")
        print(f"Duration: {analysis['analysis_duration_seconds']} seconds")
        print(f"Channels: {analysis['parameters']['channels_analyzed']}")
```
