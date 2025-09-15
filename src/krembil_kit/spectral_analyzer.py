"""
SpectralAnalyzer - Comprehensive EEG Spectral Analysis Tool
==========================================================

This module provides advanced spectral analysis capabilities for EEG signals,
offering both time-domain power analysis and frequency-domain parameterization
methods for comprehensive neural signal characterization.

Features:
- Multi-band power analysis with configurable frequency bands and smoothing
- Spectral parameterization using FOOOF/SpecParam methodology
- Advanced signal preprocessing and filtering capabilities
- Professional visualization and data export functionality
- Support for trigger-based signal segmentation
- Batch processing across multiple EEG channels
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend, butter, filtfilt, iirnotch
from scipy.stats import trim_mean
from pathlib import Path
import datetime
import time

from specparam import SpectralModel as FOOOF


class SpectralAnalyzer:
    """
    Advanced spectral analysis tool for EEG signal processing and characterization.
    
    The SpectralAnalyzer provides comprehensive frequency-domain analysis capabilities
    for electrophysiological data, implementing modern methods for neural
    signal decomposition and quantification.
    
    Analysis Methods:
        Multi-band Power Analysis:
            - Canonical EEG frequency band decomposition (Delta, Theta, Alpha, Beta, Gamma)
            - Configurable bandpass filtering with Butterworth filters
            - Signal rectification and moving-average smoothing
            - Time-series power estimation with multiple smoothing windows
        
        Spectral Parameterization:
            - Separation of aperiodic (1/f) and periodic (oscillatory) components
            - FOOOF/SpecParam model fitting for neural power spectra
            - Peak detection and characterization in frequency domain
            - Model validation with goodness-of-fit metrics
    
    Key Features:
        - Professional-grade signal preprocessing pipeline
        - Trigger-based signal segmentation for event-related analysis
        - Batch processing across multiple EEG channels
        - Comprehensive visualization suite with publication-ready plots
        - Structured data export in CSV and JSON formats
        - Configurable analysis parameters for research flexibility
    """
    
    def __init__(self, loader, trigger_detector=None, target_length: int = 50, 
                 output_dir: str = None):
        """
        Initialize the SpectralAnalyzer with EEG data and analysis parameters.
        
        Parameters
        ----------
        loader : EDFLoader
            Configured EDFLoader instance containing loaded EEG signals.
            Must have signals loaded via load_and_plot_signals() method.
        trigger_detector : TriggerDetector, optional
            TriggerDetector instance for event-based signal segmentation.
            Required for multi-band power analysis with temporal segmentation.
        target_length : int, default=50
            Number of resampled points per segment for temporal aggregation.
            Controls the resolution of time-series outputs.
        output_dir : str, optional
            Output directory path for analysis results. If None, defaults to 
            'spectral_analysis_results' subdirectory in the same directory as the EDF file.
        
        Raises
        ------
        ValueError
            If the loader contains no loaded signals.
        """
        self.loader = loader
        self.trigger_detector = trigger_detector
        self.df_triggers = trigger_detector.df_triggers if trigger_detector else None
        
        # Validate that signals are loaded
        if not self.loader.signals_dict:
            raise ValueError("No signals loaded in EDFLoader. Call load_and_plot_signals() first.")
        
        # Use channels from the loaded signals
        self.channels = list(loader.signals_dict.keys())
        
        # Configure output directory and subject name
        self.subject_name = loader.name
        if output_dir is None:
            loader_path = Path(self.loader.folder_path) / self.loader.name
            self.output_dir = loader_path / "spectral_analysis_results"
        else:
            self.output_dir = Path(output_dir)
        
        # Standard EEG channel reference for full recordings
        self.standard_eeg_channels = [
            'Fp1', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz'
        ]
        
        # Canonical EEG frequency bands
        self.frequency_bands = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 12),
            "Beta": (12, 30),
            "Gamma": (30, 80)
        }
        
        # Configuration for multi-band analysis
        self.target_length = target_length
        self.smoothing_window_secs = [0.10, 0.25, 0.50]  # Moving-average windows in seconds
        
        # Configuration for spectral parameterization (FOOOF)
        self.fooof_settings = {
            'peak_width_limits': (1, 8),
            'max_n_peaks': 6,
            'min_peak_height': 0.1,
            'peak_threshold': 2.0,
            'aperiodic_mode': 'fixed',
            'verbose': False
        }
        self.fooof_freq_range = (1, 40)
        self.psd_nperseg = 1024
        
        # Storage for analysis results
        self.multiband_results = {}
        self.fooof_results = {}
        
        # Library information for spectral parameterization
        self._library_info = self._get_spectral_param_library_info()

    def _save_analysis_metadata(self, analysis_type: str, parameters: dict, results_info: dict, 
                               analysis_start_time: float, analysis_end_time: float):
        """Save analysis metadata to JSON file, appending to existing entries."""
        metadata_file = self.output_dir / "analysis_metadata.json"
        
        # Create new metadata entry
        new_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_type": analysis_type,
            "analysis_duration_seconds": round(analysis_end_time - analysis_start_time, 2),
            "parameters": parameters,
            "data_info": {
                "loader_folder_path": str(self.loader.folder_path),
                "subject_name": self.subject_name,
                "channels": self.channels
            },
            "results": results_info
        }
        
        # Load existing metadata or create new list
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata_list = json.load(f)
                if not isinstance(metadata_list, list):
                    metadata_list = [metadata_list]  # Convert old single-entry format
            except (json.JSONDecodeError, FileNotFoundError):
                metadata_list = []
        else:
            metadata_list = []
        
        # Append new entry
        metadata_list.append(new_entry)
        
        # Save updated metadata
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
    
    # ========================================================================
    # PUBLIC METHODS - Multi-band Power Analysis
    # ========================================================================
    
    def analyze_multiband_power(self, channels_to_analyze=None):
        """
        Execute comprehensive multi-band power analysis across EEG frequency bands.
        
        Implements a robust signal processing pipeline for time-domain power estimation:
        
        1. **Frequency Band Decomposition**: Applies Butterworth bandpass filters
           for canonical EEG bands (Delta: 0.5-4Hz, Theta: 4-8Hz, Alpha: 8-12Hz,
           Beta: 12-30Hz, Gamma: 30-80Hz).
        
        2. **Power Estimation**: Computes instantaneous power via signal rectification
           (absolute value transformation).
        
        3. **Temporal Smoothing**: Applies configurable moving-average windows
           to reduce noise and enhance signal-to-noise ratio.
        
        4. **Data Export**: Generates structured CSV outputs and publication-ready
           visualizations for each frequency band and smoothing configuration.
        
        Parameters
        ----------
        channels_to_analyze : list of str, optional
            EEG channel names to process. If None, processes all loaded channels.
            Channel names must match those available in the loaded dataset.
        
        Raises
        ------
        ValueError
            If no trigger detector is provided (required for segmented analysis).
        
        Notes
        -----
        Results are saved to organized directory structure:
        - CSV files: `{base_dir}/{band}/csv/{channel}_{band}_ma{window}ms_median.csv`
        - Plots: `{base_dir}/{band}/plots/{channel}_{band}_multiband_analysis.png`
        """
        analysis_start_time = time.time()
        
        if self.df_triggers is None or self.df_triggers.empty:
            raise ValueError("Trigger detector required for multi-band analysis. Please provide trigger_detector in constructor.")
        
        channels_to_analyze = self._validate_channels(channels_to_analyze)
        if not channels_to_analyze:
            return
        
        print(f"Starting multi-band power analysis on {len(channels_to_analyze)} channels: {channels_to_analyze}")
        
        for channel in channels_to_analyze:
            print(f"Processing channel: {channel}")
            self._process_channel_multiband(channel)
        
        print("Multi-band power analysis complete.")
        
        # Save analysis metadata
        analysis_end_time = time.time()
        parameters = {
            "channels_analyzed": channels_to_analyze,
            "frequency_bands": self.frequency_bands,
            "target_length": self.target_length,
            "smoothing_window_secs": self.smoothing_window_secs
        }
        results_info = {
            "analysis_type": "multiband_power",
            "channels_processed": len(channels_to_analyze),
            "frequency_bands_analyzed": list(self.frequency_bands.keys())
        }
        self._save_analysis_metadata("multiband_power", parameters, results_info, 
                                   analysis_start_time, analysis_end_time)
    
    def analyze_spectral_parameterization(self, channels_to_analyze=None):
        """
        Execute advanced spectral parameterization using FOOOF/SpecParam methodology.
        
        Implements state-of-the-art spectral decomposition to separate neural power
        spectra into distinct physiological components:
        
        **Aperiodic Component**: Characterizes the broadband 1/f background activity
        reflecting neural population dynamics and cortical excitation/inhibition balance.
        
        **Periodic Components**: Identifies and quantifies discrete oscillatory peaks
        corresponding to rhythmic neural activity (e.g., alpha, beta oscillations).
        
        The analysis pipeline includes:
        - Advanced signal preprocessing (detrending, filtering, artifact removal)
        - Power spectral density estimation via Welch's method
        - Robust model fitting with configurable parameters
        - Comprehensive validation metrics and goodness-of-fit assessment
        - Frequency band power quantification
        
        Parameters
        ----------
        channels_to_analyze : list of str, optional
            EEG channel names to process. If None, processes all loaded channels.
            Channel names must match those available in the loaded dataset.
        
        Notes
        -----
        Results are organized in structured output directory:
        - Individual channel results: `{output_dir}/{channel}/`
        - Model fit visualizations: `{channel}_fooof_model_fit.png`
        - Parameter summaries: `{channel}_fooof_parameters.csv`
        - Band power estimates: `{channel}_frequency_band_powers.csv`
        - Cross-channel summaries: `fooof_parameters_summary.csv`
        
        The method automatically detects and uses the optimal spectral parameterization
        library (SpecParam preferred, FOOOF fallback for compatibility).
        """
        analysis_start_time = time.time()
        
        channels_to_analyze = self._validate_channels(channels_to_analyze)
        if not channels_to_analyze:
            return
        
        print(f"Starting spectral parameterization analysis on {len(channels_to_analyze)} channels...")
        print(f"Using {self._library_info['library']} v{self._library_info['version']} ({self._library_info['description']})")
        
        # Create output directory
        output_dir = self.output_dir / 'spectral_parameterization'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each channel
        for channel in channels_to_analyze:
            print(f"Processing channel: {channel}")
            result = self._process_channel_fooof(channel)
            self.fooof_results[channel] = result
            self._save_fooof_channel_results(result, output_dir)
        
        # Save summary results
        self._save_fooof_summary_results(output_dir)
        print(f"Spectral parameterization analysis complete. Results saved to: {output_dir}")
        
        # Save analysis metadata
        analysis_end_time = time.time()
        parameters = {
            "channels_analyzed": channels_to_analyze,
            "fooof_settings": self.fooof_settings,
            "fooof_freq_range": self.fooof_freq_range,
            "psd_nperseg": self.psd_nperseg
        }
        results_info = {
            "analysis_type": "spectral_parameterization",
            "channels_processed": len(channels_to_analyze),
            "output_directory": str(output_dir),
            "library_info": self._library_info
        }
        self._save_analysis_metadata("spectral_parameterization", parameters, results_info, 
                                   analysis_start_time, analysis_end_time)
    
    def analyze_comprehensive(self, channels_to_analyze=None):
        """
        Execute complete spectral analysis suite combining all available methods.
        
        Performs sequential execution of both time-domain multi-band power analysis
        and frequency-domain spectral parameterization, providing comprehensive
        characterization of neural signal properties across temporal and spectral domains.
        
        This integrated approach enables:
        - Complete frequency band characterization
        - Temporal dynamics quantification
        - Aperiodic/periodic component separation
        - Cross-method validation and comparison
        
        Parameters
        ----------
        channels_to_analyze : list of str, optional
            EEG channel names to process. If None, processes all loaded channels.
        
        Notes
        -----
        Multi-band analysis requires trigger detector for temporal segmentation.
        If no trigger detector is provided, only spectral parameterization is performed.
        """
        analysis_start_time = time.time()
        
        print("Starting comprehensive spectral analysis...")
        
        if self.df_triggers is not None and not self.df_triggers.empty:
            print("\n=== Multi-band Power Analysis ===")
            self.analyze_multiband_power(channels_to_analyze)
        else:
            print("Skipping multi-band analysis (no trigger detector provided)")
        
        print("\n=== Spectral Parameterization Analysis ===")
        self.analyze_spectral_parameterization(channels_to_analyze)
        
        print("\nComprehensive spectral analysis complete.")
        
        # Save analysis metadata
        analysis_end_time = time.time()
        parameters = {
            "channels_analyzed": channels_to_analyze,
            "multiband_enabled": self.df_triggers is not None and not self.df_triggers.empty,
            "spectral_parameterization_enabled": True
        }
        results_info = {
            "analysis_type": "comprehensive",
            "methods_executed": ["spectral_parameterization"] + (["multiband_power"] if self.df_triggers is not None and not self.df_triggers.empty else [])
        }
        self._save_analysis_metadata("comprehensive", parameters, results_info, 
                                   analysis_start_time, analysis_end_time)
    
    # ========================================================================
    # PUBLIC METHODS - Visualization
    # ========================================================================
    
    def plot_raw_signal_window(self, window_index: int, channel: str):
        """
        Generate publication-ready visualization of raw EEG data for specified window.
        
        Creates high-quality time-series plot of unprocessed EEG signal within
        a trigger-defined temporal window, with professional formatting suitable
        for research publications and presentations.
        
        Parameters
        ----------
        window_index : int
            Zero-based index of the trigger-defined window to visualize.
            Must be less than (total_triggers - 1) to ensure valid window bounds.
        channel : str
            EEG channel name to plot. Must exist in loaded dataset.
        
        Raises
        ------
        ValueError
            If trigger detector is not provided or window_index is out of range.
        """
        if self.df_triggers is None or self.df_triggers.empty:
            raise ValueError("Trigger detector required for window plotting.")
        
        try:
            if window_index >= len(self.df_triggers) - 1:
                raise ValueError("window_index out of range – needs a subsequent trigger.")
            
            start_idx = int(self.df_triggers.iloc[window_index]['end_index'])
            end_idx = int(self.df_triggers.iloc[window_index + 1]['start_index'])
            
            signal = self.loader.signals_dict[channel]['data']
            fs = self.loader.signals_dict[channel]['sample_rate']
            
            time_minutes = np.arange(end_idx - start_idx) / fs / 60  # Convert to minutes
            
            plt.figure(figsize=(12, 6))
            plt.plot(time_minutes, signal[start_idx:end_idx])
            plt.title(f'Raw Signal Window {window_index} | Channel: {channel}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Amplitude (µV)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as exc:
            print(f"Error plotting window {window_index} for {channel}: {exc}")
    
    def plot_averaged_signal_window(self, channel: str, start_window: int = None,
                                   end_window: int = None, target_length: int = 500,
                                   aggregation_method: str = 'mean', trim_ratio: float = 0.1):
        """
        Generate ensemble-averaged signal visualization across multiple temporal windows.
        
        Creates publication-quality visualization of aggregated EEG signal across
        specified trigger-defined windows, implementing robust statistical aggregation
        methods to enhance signal-to-noise ratio and reveal consistent temporal patterns.
        
        Parameters
        ----------
        channel : str
            EEG channel name to analyze and visualize.
        start_window : int, optional
            Starting window index for aggregation. If None, uses first window (0).
        end_window : int, optional
            Ending window index for aggregation. If None, uses last available window.
        target_length : int, default=500
            Number of temporal points for signal resampling and standardization.
            Higher values provide better temporal resolution.
        aggregation_method : {'mean', 'median', 'trimmed'}, default='mean'
            Statistical method for cross-window aggregation:
            - 'mean': Arithmetic mean across windows
            - 'median': Robust median aggregation
            - 'trimmed': Trimmed mean excluding outlier windows
        trim_ratio : float, default=0.1
            Proportion of extreme values to exclude for 'trimmed' aggregation method.
            Range: 0.0 to 0.5, where 0.1 excludes 10% of extreme values.
        
        Raises
        ------
        ValueError
            If trigger detector is not provided or aggregation_method is invalid.
        """
        if self.df_triggers is None or self.df_triggers.empty:
            raise ValueError("Trigger detector required for averaged window plotting.")
        
        if start_window is None:
            start_window = 0
        if end_window is None:
            end_window = len(self.df_triggers) - 1
        
        signal = self.loader.signals_dict[channel]['data']
        fs = self.loader.signals_dict[channel]['sample_rate']
        segments, durations = [], []
        
        # Extract and resample segments
        for i in range(start_window, min(end_window, len(self.df_triggers) - 1)):
            start_idx = int(self.df_triggers.iloc[i]['end_index'])
            end_idx = int(self.df_triggers.iloc[i + 1]['start_index'])
            
            if end_idx <= start_idx:
                continue
            
            segment = signal[start_idx:end_idx]
            segments.append(self._resample_signal(segment, target_length))
            durations.append((end_idx - start_idx) / fs)
        
        if not segments:
            print("No valid windows in requested range.")
            return
        
        # Aggregate segments
        stacked_segments = np.stack(segments)
        if aggregation_method == 'mean':
            aggregated = np.mean(stacked_segments, axis=0)
        elif aggregation_method == 'median':
            aggregated = np.median(stacked_segments, axis=0)
        elif aggregation_method == 'trimmed':
            aggregated = np.array([trim_mean(stacked_segments[:, i], trim_ratio) 
                                 for i in range(stacked_segments.shape[1])])
        else:
            raise ValueError("aggregation_method must be 'mean', 'median', or 'trimmed'.")
        
        # Create time axis in minutes
        time_axis_minutes = np.linspace(0, np.mean(durations) / 60, target_length)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis_minutes, aggregated, linewidth=2)
        plt.title(f'Aggregated Raw Signal ({aggregation_method.title()}) | Channel: {channel}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Amplitude (µV)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_fooof_comparison(self, channels=None, metric='aperiodic_exponent'):
        """
        Generate comparative visualization of spectral parameterization metrics across channels.
        
        Creates professional bar chart comparing specified spectral parameterization
        metrics across EEG channels, enabling identification of spatial patterns
        in neural oscillatory and aperiodic activity.
        
        Parameters
        ----------
        channels : list of str, optional
            EEG channel names to include in comparison. If None, includes all
            channels with completed spectral parameterization analysis.
        metric : {'aperiodic_exponent', 'aperiodic_offset', 'n_peaks', 'r_squared', 'error'}, default='aperiodic_exponent'
            Spectral parameterization metric to visualize:
            - 'aperiodic_exponent': 1/f slope reflecting neural population dynamics
            - 'aperiodic_offset': Broadband power offset parameter
            - 'n_peaks': Number of detected oscillatory peaks
            - 'r_squared': Model fit quality (coefficient of determination)
            - 'error': Root mean square error of model fit
        
        Notes
        -----
        Requires prior execution of analyze_spectral_parameterization() method.
        Visualization includes professional formatting with grid lines, proper
        axis labels, and publication-ready styling.
        """
        if not self.fooof_results:
            print("No FOOOF results available. Run analyze_spectral_parameterization() first.")
            return
        
        if channels is None:
            channels = list(self.fooof_results.keys())
        
        values, labels = [], []
        
        for channel in channels:
            if channel in self.fooof_results:
                result = self.fooof_results[channel]
                
                if metric == 'aperiodic_exponent':
                    values.append(result['aperiodic_params'][1])
                elif metric == 'aperiodic_offset':
                    values.append(result['aperiodic_params'][0])
                elif metric == 'n_peaks':
                    values.append(len(result['peak_params']))
                elif metric == 'r_squared':
                    values.append(result['r_squared'])
                elif metric == 'error':
                    values.append(result['error'])
                else:
                    print(f"Unknown metric: {metric}")
                    return
                
                labels.append(channel)
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values, color='steelblue', alpha=0.7)
        plt.title(f'{metric.replace("_", " ").title()} Across Channels')
        plt.xlabel('Channel')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # PUBLIC METHODS - Configuration
    # ========================================================================
    
    def set_frequency_bands(self, bands_dict):
        """
        Configure custom frequency bands for multi-band power analysis.
        
        Allows specification of application-specific frequency bands beyond
        the canonical EEG bands, enabling targeted analysis of specific
        neural oscillations or research-specific frequency ranges.
        
        Parameters
        ----------
        bands_dict : dict
            Dictionary mapping band names to frequency ranges.
            Format: {band_name: (low_freq_hz, high_freq_hz)}
            Example: {'slow_alpha': (8, 10), 'fast_alpha': (10, 12)}
        
        Notes
        -----
        Custom bands replace the default canonical EEG bands for subsequent
        multi-band power analysis. Frequency ranges should not overlap
        significantly to ensure meaningful decomposition.
        """
        self.frequency_bands = bands_dict.copy()
        print(f"Updated frequency bands: {self.frequency_bands}")
    
    def set_fooof_parameters(self, freq_range=None, **fooof_kwargs):
        """
        Configure spectral parameterization analysis parameters.
        
        Provides fine-grained control over FOOOF/SpecParam model fitting
        parameters to optimize analysis for specific research applications
        or signal characteristics.
        
        Parameters
        ----------
        freq_range : tuple of float, optional
            Frequency range for analysis as (low_freq_hz, high_freq_hz).
            Default: (1, 40) Hz covering typical EEG spectrum.
        **fooof_kwargs : dict
            Additional FOOOF/SpecParam parameters:
            - peak_width_limits : tuple, peak bandwidth constraints
            - max_n_peaks : int, maximum number of peaks to detect
            - min_peak_height : float, minimum peak height threshold
            - peak_threshold : float, peak detection threshold
            - aperiodic_mode : str, aperiodic fitting mode ('fixed' or 'knee')
        
        Notes
        -----
        Parameter changes affect all subsequent spectral parameterization analyses.
        Refer to FOOOF/SpecParam documentation for detailed parameter descriptions
        and recommended values for different applications.
        """
        if freq_range is not None:
            self.fooof_freq_range = freq_range
        
        self.fooof_settings.update(fooof_kwargs)
        print(f"Updated FOOOF settings: {self.fooof_settings}")
        print(f"Frequency range: {self.fooof_freq_range}")
    
    def set_smoothing_windows(self, window_secs_list):
        """
        Configure temporal smoothing parameters for multi-band power analysis.
        
        Sets the moving-average window sizes used for temporal smoothing
        of rectified band-limited signals, allowing optimization of the
        trade-off between temporal resolution and noise reduction.
        
        Parameters
        ----------
        window_secs_list : list of float
            Moving-average window sizes in seconds. Multiple windows enable
            comparison of different temporal smoothing scales.
            Example: [0.1, 0.25, 0.5] for 100ms, 250ms, and 500ms windows.
        
        Notes
        -----
        Shorter windows preserve temporal dynamics but may be noisier.
        Longer windows provide smoother estimates but reduce temporal resolution.
        Window sizes are automatically converted to samples based on signal
        sampling frequency during analysis.
        """
        self.smoothing_window_secs = window_secs_list.copy()
        print(f"Updated smoothing windows: {self.smoothing_window_secs} seconds")
    
    def get_analysis_info(self):
        """
        Retrieve comprehensive analysis configuration and system information.
        
        Provides detailed information about current analyzer settings,
        available channels, and system capabilities for documentation
        and reproducibility purposes.
        
        Returns
        -------
        dict
            Configuration dictionary containing:
            - channels : list, available EEG channel names
            - frequency_bands : dict, configured frequency band definitions
            - smoothing_windows_sec : list, temporal smoothing window sizes
            - fooof_settings : dict, spectral parameterization parameters
            - fooof_freq_range : tuple, frequency range for parameterization
            - spectral_param_library : dict, library version information
            - has_triggers : bool, trigger detector availability status
            - target_length : int, resampling target length
        
        Notes
        -----
        This information is essential for method documentation and ensuring
        reproducible analysis across different datasets and research groups.
        """
        return {
            'channels': self.channels,
            'frequency_bands': self.frequency_bands,
            'smoothing_windows_sec': self.smoothing_window_secs,
            'fooof_settings': self.fooof_settings,
            'fooof_freq_range': self.fooof_freq_range,
            'spectral_param_library': self._library_info,
            'has_triggers': self.df_triggers is not None,
            'target_length': self.target_length
        }
    
    # ========================================================================
    # PRIVATE METHODS - Signal Processing Utilities
    # ========================================================================
    
    def _validate_channels(self, channels_to_analyze):
        """
        Validate and filter requested channels against available dataset channels.
        
        Internal method for ensuring requested channels exist in loaded dataset
        and providing informative feedback about channel availability.
        """
        if channels_to_analyze is None:
            return self.channels
        
        available_channels = set(self.channels)
        requested_channels = set(channels_to_analyze)
        missing_channels = requested_channels - available_channels
        
        if missing_channels:
            print(f"Warning: Channels {missing_channels} not available. Available: {available_channels}")
        
        valid_channels = list(requested_channels & available_channels)
        if not valid_channels:
            print("No valid channels to analyze.")
        
        return valid_channels
    
    def _resample_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """
        Perform linear interpolation resampling of temporal signal data.
        
        Internal utility for standardizing signal lengths across analysis windows
        while preserving temporal characteristics through linear interpolation.
        """
        if signal.size == 0:
            return np.array([])
        
        old_indices = np.linspace(0, signal.size - 1, num=signal.size)
        new_indices = np.linspace(0, signal.size - 1, num=target_length)
        return np.interp(new_indices, old_indices, signal)
    
    def _apply_bandpass_filter(self, signal: np.ndarray, low_freq: float, 
                              high_freq: float, fs: float, order: int = 4) -> np.ndarray:
        """
        Apply zero-phase Butterworth bandpass filter for frequency band isolation.
        
        Internal method implementing high-quality digital filtering with zero-phase
        distortion using forward-backward filtering (filtfilt) for optimal
        frequency selectivity and temporal preservation.
        """
        nyquist = 0.5 * fs
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        b, a = butter(order, [low_norm, high_norm], btype='band')
        return filtfilt(b, a, signal)
    
    def _apply_moving_average(self, signal: np.ndarray, window_samples: int) -> np.ndarray:
        """
        Apply temporal smoothing via convolution-based moving average filtering.
        
        Internal utility for noise reduction and temporal smoothing of power
        estimates using efficient convolution with uniform kernel.
        """
        if window_samples <= 1:
            return signal
        
        kernel = np.ones(window_samples) / window_samples
        return np.convolve(signal, kernel, mode='same')
    
    def _preprocess_signal_for_fooof(self, signal, fs, detrend_signal=True,
                                    bandpass=(1, 40), notch_freq=60.0, notch_quality=30.0):
        """
        Execute comprehensive signal preprocessing pipeline for spectral parameterization.
        
        Internal method implementing multi-stage preprocessing optimized for
        spectral parameterization analysis, including detrending, filtering,
        and artifact removal to ensure high-quality power spectral density estimation.
        """
        processed = signal.copy()
        
        # Detrend signal
        if detrend_signal:
            processed = detrend(processed)
        
        # Apply bandpass filter
        if bandpass is not None:
            low, high = bandpass
            processed = self._apply_bandpass_filter(processed, low, high, fs)
        
        # Apply notch filter (typically for line noise)
        if notch_freq is not None:
            nyquist = fs / 2
            w0 = notch_freq / nyquist
            b, a = iirnotch(w0, notch_quality)
            processed = filtfilt(b, a, processed)
        
        return processed
    
    def _compute_power_spectral_density(self, signal, fs):
        """
        Estimate power spectral density using Welch's periodogram method.
        
        Internal method for robust frequency-domain power estimation with
        optimal windowing and overlap parameters for neural signal analysis.
        """
        frequencies, psd = welch(signal, fs=fs, nperseg=self.psd_nperseg)
        return frequencies, psd
    
    def _calculate_band_powers(self, frequencies, psd, bands=None):
        """
        Quantify spectral power within defined frequency bands via numerical integration.
        
        Internal method for computing band-limited power estimates using
        trapezoidal integration over specified frequency ranges.
        """
        if bands is None:
            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 40)
            }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies <= high)
            if np.any(mask):
                power = np.trapezoid(psd[mask], frequencies[mask])
                band_powers[band_name] = power
            else:
                band_powers[band_name] = 0.0
        
        return band_powers
    
    def _get_spectral_param_library_info(self):
        """
        Detect and characterize available spectral parameterization library.
        
        Internal method for automatic detection of FOOOF/SpecParam library
        availability and version information for compatibility management.
        """
        try:
            import specparam
            return {
                'library': 'specparam',
                'version': getattr(specparam, '__version__', 'unknown'),
                'description': 'Modern spectral parameterization library'
            }
        except ImportError:
            try:
                import fooof
                return {
                    'library': 'fooof',
                    'version': getattr(fooof, '__version__', 'unknown'),
                    'description': 'Legacy FOOOF library (compatibility mode)'
                }
            except ImportError:
                return {
                    'library': 'none',
                    'version': 'unknown',
                    'description': 'No spectral parameterization library found'
                }    

    # ========================================================================
    # PRIVATE METHODS - Multi-band Analysis Processing
    # ========================================================================
    
    def _process_channel_multiband(self, channel):
        """
        Execute complete multi-band power analysis pipeline for single EEG channel.
        
        Internal method coordinating frequency band decomposition, power estimation,
        temporal smoothing, and result aggregation for individual channel processing.
        """
        channel_data = self.loader.signals_dict[channel]
        signal = channel_data['data']
        fs = channel_data['sample_rate']
        
        # Convert smoothing windows from seconds to samples
        ma_windows_samples = [int(fs * window_sec) for window_sec in self.smoothing_window_secs]
        durations = []  # Track segment durations
        
        # Pre-allocate results storage: {band -> {window -> list[segments]}}
        band_results = {
            band: {window: [] for window in ma_windows_samples} 
            for band in self.frequency_bands
        }
        
        # Process each trigger-defined segment
        for i in range(len(self.df_triggers) - 1):
            start_idx = int(self.df_triggers.iloc[i]['end_index'])
            end_idx = int(self.df_triggers.iloc[i + 1]['start_index'])
            
            # Skip segments that are too short
            if end_idx - start_idx < 2 * fs:
                continue
            
            segment = signal[start_idx:end_idx]
            duration_sec = (end_idx - start_idx) / fs
            durations.append(duration_sec)
            
            # Process each frequency band
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Apply bandpass filter
                filtered = self._apply_bandpass_filter(segment, low_freq, high_freq, fs)
                
                # Rectify (absolute value as power proxy)
                rectified = np.abs(filtered)
                
                # Apply different moving average windows
                for window_samples in ma_windows_samples:
                    smoothed = self._apply_moving_average(rectified, window_samples)
                    resampled = self._resample_signal(smoothed, self.target_length)
                    band_results[band_name][window_samples].append(resampled)
        
        if not durations:
            print(f"No valid segments found for channel {channel}.")
            return
        
        # Save results and create plots
        self._save_multiband_results(channel, band_results, ma_windows_samples, 
                                   durations, fs)
    
    def _save_multiband_results(self, channel, band_results, ma_windows_samples, 
                               durations, fs):
        """
        Export multi-band analysis results with structured file organization.
        
        Internal method for generating publication-ready outputs including
        CSV data files and professional visualizations with consistent
        formatting and directory structure.
        """
        avg_duration_sec = np.mean(durations)
        time_axis_minutes = np.linspace(0, avg_duration_sec / 60, self.target_length)
        
        # Create unified output directories
        csv_dir = self.output_dir / "multiband_power" / "csv"
        plot_dir = self.output_dir / "multiband_power" / "plots"
        csv_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each frequency band
        for band_name in self.frequency_bands:
            plt.figure(figsize=(12, 8))
            
            for window_samples in ma_windows_samples:
                segments = band_results[band_name][window_samples]
                if not segments:
                    continue
                
                # Calculate median across segments
                median_signal = np.median(np.stack(segments), axis=0)
                window_ms = int(1000 * window_samples / fs)
                
                # Save to CSV with new naming convention
                csv_filename = f"{self.subject_name}_multiband_{band_name}_ma{window_ms}ms.csv"
                csv_path = csv_dir / csv_filename
                
                df = pd.DataFrame({
                    f"{channel}_{band_name}_ma{window_ms}ms": median_signal,
                    'time_minutes': time_axis_minutes
                })
                df.to_csv(csv_path, index=False)
                print(f"Saved: {csv_path}")
                
                # Add to plot
                plt.plot(time_axis_minutes, median_signal, 
                        label=f"MA {window_ms} ms", linewidth=2)
            
            # Finalize and save plot with new naming convention
            plt.title(f'Median {band_name}-band Power | Channel: {channel}', fontsize=14)
            plt.xlabel('Time (minutes)', fontsize=12)
            plt.ylabel('Power (arbitrary units)', fontsize=12)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_filename = f"{self.subject_name}_multiband_{band_name}_{channel}.png"
            plot_path = plot_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Plot saved: {plot_path}")
    
    # ========================================================================
    # PRIVATE METHODS - FOOOF Analysis Processing
    # ========================================================================
    
    def _process_channel_fooof(self, channel):
        """
        Execute complete spectral parameterization analysis for single EEG channel.
        
        Internal method coordinating signal preprocessing, power spectral density
        estimation, FOOOF model fitting, and result compilation for individual
        channel processing.
        """
        signal_data = self.loader.signals_dict[channel]['data']
        fs = self.loader.signals_dict[channel]['sample_rate']
        
        # Preprocess signal
        processed_signal = self._preprocess_signal_for_fooof(signal_data, fs)
        
        # Compute power spectral density
        frequencies, psd = self._compute_power_spectral_density(processed_signal, fs)
        
        # Initialize and fit FOOOF model
        fooof_model = FOOOF(**self.fooof_settings)
        fooof_model.fit(frequencies, psd, self.fooof_freq_range)
        
        # Compile results
        results = {
            'channel': channel,
            'aperiodic_params': fooof_model.aperiodic_params_,
            'peak_params': fooof_model.peak_params_,
            'r_squared': fooof_model.r_squared_,
            'error': fooof_model.error_,
            'frequencies': frequencies,
            'psd': psd,
            'fooof_model': fooof_model
        }
        
        return results
    
    def _save_fooof_channel_results(self, result, output_dir):
        """
        Export spectral parameterization results with comprehensive documentation.
        
        Internal method for generating structured output files including model
        visualizations, parameter tables, and configuration documentation
        for individual channel analysis results.
        """
        channel = result['channel']
        
        # Create organized subdirectories
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FOOOF model fit plot with new naming convention
        fig_path = individual_dir / f'{self.subject_name}_fooof_{channel}.png'
        result['fooof_model'].plot()
        plt.title(f'FOOOF Model Fit - Channel: {channel}', fontsize=14)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save power spectral density plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(result['frequencies'], result['psd'], 
                    label='Original PSD', color='steelblue', linewidth=2)
        plt.xlim(self.fooof_freq_range)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Power (log scale)', fontsize=12)
        plt.title(f'Power Spectral Density - Channel: {channel}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        psd_path = individual_dir / f'{self.subject_name}_psd_{channel}.png'
        plt.savefig(psd_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save FOOOF parameters to CSV with new naming convention
        params_data = {
            'channel': [channel],
            'aperiodic_offset': [result['aperiodic_params'][0]],
            'aperiodic_exponent': [result['aperiodic_params'][1]],
            'r_squared': [result['r_squared']],
            'error': [result['error']],
            'n_peaks': [len(result['peak_params'])]
        }
        
        # Add peak parameters if peaks were detected
        if len(result['peak_params']) > 0:
            for i, peak in enumerate(result['peak_params']):
                params_data[f'peak_{i+1}_frequency'] = [peak[0]]
                params_data[f'peak_{i+1}_power'] = [peak[1]]
                params_data[f'peak_{i+1}_bandwidth'] = [peak[2]]
        
        params_df = pd.DataFrame(params_data)
        params_path = individual_dir / f'{self.subject_name}_fooof_parameters_{channel}.csv'
        params_df.to_csv(params_path, index=False)
        
        # Save band powers
        band_powers = self._calculate_band_powers(result['frequencies'], result['psd'])
        band_df = pd.DataFrame([band_powers])
        band_df['channel'] = channel
        band_path = individual_dir / f'{self.subject_name}_band_powers_{channel}.csv'
        band_df.to_csv(band_path, index=False)
        
        # Save FOOOF model settings
        settings_path = individual_dir / f'{self.subject_name}_fooof_settings_{channel}.json'
        with open(settings_path, 'w') as f:
            json.dump(result['fooof_model'].get_settings(), f, indent=4)
    
    def _save_fooof_summary_results(self, output_dir):
        """
        Generate cross-channel summary analysis and comparative visualizations.
        
        Internal method for creating comprehensive summary reports aggregating
        spectral parameterization results across all analyzed channels with
        statistical summaries and comparative visualizations.
        """
        if not self.fooof_results:
            return
        
        summary_data = []
        band_powers_data = []
        
        # Compile summary data for each channel
        for channel, result in self.fooof_results.items():
            # Basic FOOOF parameters
            row = {
                'channel': channel,
                'aperiodic_offset': result['aperiodic_params'][0],
                'aperiodic_exponent': result['aperiodic_params'][1],
                'r_squared': result['r_squared'],
                'error': result['error'],
                'n_peaks': len(result['peak_params'])
            }
            
            # Add dominant peak information if peaks exist
            if len(result['peak_params']) > 0:
                # Find peak with highest power
                dominant_peak = result['peak_params'][np.argmax(result['peak_params'][:, 1])]
                row['dominant_peak_frequency'] = dominant_peak[0]
                row['dominant_peak_power'] = dominant_peak[1]
                row['dominant_peak_bandwidth'] = dominant_peak[2]
            else:
                row['dominant_peak_frequency'] = np.nan
                row['dominant_peak_power'] = np.nan
                row['dominant_peak_bandwidth'] = np.nan
            
            summary_data.append(row)
            
            # Band powers
            band_powers = self._calculate_band_powers(result['frequencies'], result['psd'])
            band_powers['channel'] = channel
            band_powers_data.append(band_powers)
        
        # Create summary directory and save CSV files with new naming convention
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = summary_dir / f'{self.subject_name}_fooof_parameters_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        band_summary_df = pd.DataFrame(band_powers_data)
        band_summary_path = summary_dir / f'{self.subject_name}_band_powers_summary.csv'
        band_summary_df.to_csv(band_summary_path, index=False)
        
        # Create summary visualization plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        self._create_fooof_summary_plots(summary_df, band_summary_df, plots_dir)
    
    def _create_fooof_summary_plots(self, summary_df, band_df, output_dir):
        """
        Generate publication-ready summary visualizations for spectral parameterization.
        
        Internal method creating professional comparative plots including parameter
        distributions, model quality metrics, and frequency band power heatmaps
        with consistent styling and formatting.
        """
        # Aperiodic exponent comparison
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['channel'], summary_df['aperiodic_exponent'], 
                color='steelblue', alpha=0.7)
        plt.title('Aperiodic Exponent (1/f slope) Across Channels', fontsize=14)
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Aperiodic Exponent', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.subject_name}_aperiodic_exponent_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Number of peaks comparison
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['channel'], summary_df['n_peaks'], 
                color='forestgreen', alpha=0.7)
        plt.title('Number of Spectral Peaks Across Channels', fontsize=14)
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Number of Peaks', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.subject_name}_spectral_peaks_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Band powers heatmap
        band_columns = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_matrix = band_df[band_columns].values
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(band_matrix.T, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Power (arbitrary units)')
        plt.yticks(range(len(band_columns)), [col.title() for col in band_columns])
        plt.xticks(range(len(band_df)), band_df['channel'], rotation=45)
        plt.title('Frequency Band Powers Across Channels', fontsize=14)
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Frequency Band', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.subject_name}_band_powers_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model fit quality (R-squared) comparison
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['channel'], summary_df['r_squared'], 
                color='coral', alpha=0.7)
        plt.title('FOOOF Model Fit Quality (R²) Across Channels', fontsize=14)
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('R-squared', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.subject_name}_model_fit_quality_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()