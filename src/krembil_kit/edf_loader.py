import os
import pyedflib
import matplotlib.pyplot as plt

class EDFLoader:
    def __init__(self, folder_path, name):
        """
        Initializes the EDFLoader with a folder path where the EDF file is stored.

        :param folder_path: str, the base directory where the EDF file is stored
        :param name: str, the name of the subject or experiment (the EDF file should match this name)
        """
        self.folder_path = folder_path
        self.name = name
        self.edf_file_path = f"{folder_path}/{name}/{name}.edf"
        self.signals_dict = None
        # Check for the existence of the EDF file
        if not os.path.exists(self.edf_file_path):
            raise FileNotFoundError(f"EDF file not found: {self.edf_file_path}")

    def inspect_data(self):
        """
        Inspects the EDF file for the specified subject, printing out various signal information.
        """
        try:
            with pyedflib.EdfReader(self.edf_file_path) as edf:
                print("File header:")
                print(edf.getHeader())
                n = edf.signals_in_file
                print("\nNumber of signals:", n)

                for i in range(n):
                    print(f"\nSignal {i} information:")
                    print("Label:", edf.getLabel(i))
                    print("Sample rate:", edf.getSampleFrequency(i))
                    print("Duration:", edf.getFileDuration())
                    print("Physical maximum:", edf.getPhysicalMaximum(i))
                    print("Physical minimum:", edf.getPhysicalMinimum(i))
                    print("Digital maximum:", edf.getDigitalMaximum(i))
                    print("Digital minimum:", edf.getDigitalMinimum(i))
                    print("First 10 samples:", edf.readSignal(i)[:10])
        except Exception as e:
            print(f"An error occurred: {e}")

    def load_and_plot_signals(self, signal_indices=None, duration=None, save_plots=False, save_path=None):
        """
        Loads and plots the signals from the EDF file, storing them along with their sample rates in a dictionary.
        
        :param signal_indices: list of int, specific signal indices to load (None = load all signals)
        :param duration: float, duration in seconds to plot (None = entire duration)
        :param save_plots: bool, whether to save plots instead of showing them (default: False)
        :param save_path: str, directory to save plots (default: plots/{subject_name})
        """
        signals_dict = {}

        try:
            with pyedflib.EdfReader(self.edf_file_path) as edf:
                self.signals_dict = signals_dict
                n = edf.signals_in_file
                
                # Determine which signals to load
                if signal_indices is None:
                    indices_to_load = list(range(n))
                    print(f"⚠️  WARNING: Loading all {n} signals might cause crashes due to memory usage! Consider loading and visualizing a subset of signals each time.")
                    print(f"Loading all {n} signals")
                else:
                    indices_to_load = [i for i in signal_indices if 0 <= i < n]
                    print(f"Loading {len(indices_to_load)} specified signals out of {n} total")
                
                # Set up save path if needed
                if save_plots:
                    if save_path is None:
                        save_path = f"plots/{self.name}"
                    os.makedirs(save_path, exist_ok=True)
                    print(f"Plots will be saved to: {save_path}")
                
                # Load selected signals
                signal_data = []
                sample_rates = []
                signal_labels = []
                
                for i in indices_to_load:
                    data = edf.readSignal(i)
                    rate = edf.getSampleFrequency(i)
                    label = edf.getLabel(i)
                    
                    # Apply duration limit if specified
                    if duration is not None:
                        max_samples = int(duration * rate)
                        if len(data) > max_samples:
                            data = data[:max_samples]
                            print(f"Truncated signal {label} to {duration} seconds ({max_samples} samples)")
                    
                    signal_data.append(data)
                    sample_rates.append(rate)
                    signal_labels.append(label)
                    signals_dict[label] = {'data': data, 'sample_rate': rate}

                # Plot the loaded signals
                num_signals = len(signal_data)
                fig, axes = plt.subplots(nrows=num_signals, ncols=1, figsize=(12, 2 * num_signals))
                if num_signals == 1:
                    axes = [axes]  # Make it iterable if there's only one signal

                for i, (data, rate, label, ax) in enumerate(zip(signal_data, sample_rates, signal_labels, axes)):
                    # Create time axis for better visualization
                    time_axis = [j / rate for j in range(len(data))]
                    ax.plot(time_axis, data)
                    ax.set_title(f'Signal {indices_to_load[i]}: {label}')
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Amplitude')

                plt.tight_layout()
                
                # Save or show plots
                if save_plots:
                    plot_filename = f"{save_path}/signals_plot.png"
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"✅ Plot saved to: {plot_filename}")
                    plt.close()  # Close to free memory
                else:
                    plt.show()

        except FileNotFoundError:
            print(f"File not found: {self.edf_file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

