import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import cv2
import os
import re
import torch, joblib, re
from PIL import Image
from torchvision.models import resnet50
import torchvision.transforms as T


class TriggerDetector:
    def __init__(self, edf_loader, signal_choice):
        """
        Initializes the TriggerDetector with an EDFLoader instance and a choice of signal.

        :param edf_loader: EDFLoader, an instance of the EDFLoader class that contains the signal data
        :param signal_choice: str, the key corresponding to the desired signal in the EDFLoader's signals_dict
        """
        self.loader = edf_loader
        if signal_choice not in self.loader.signals_dict:
            raise ValueError(f"Signal {signal_choice} not found in the loaded signals.")
        self.signal_choice = signal_choice
        self.signal = self.loader.signals_dict[signal_choice]["data"]
        self.sample_rate = self.loader.signals_dict[signal_choice]["sample_rate"]
        self.threshold_value = 60  # Hardcoded threshold value for event detection

    def butterworth_filter(self, signal, cutoff=30, order=5, btype='low'):
        """
        Applies a Butterworth filter to the signal.

        :param signal: numpy.ndarray, the input signal data to filter
        :param cutoff: float, the cutoff frequency in Hz
        :param order: int, the order of the filter
        :param btype: str, the type of the filter ('low', 'high', 'bandpass', 'bandstop')
        :return: numpy.ndarray, the filtered signal
        """
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, signal)

    def detect_triggers(self):
        """
        Detects triggers in the signal based on a threshold and filters events to be between
        55 seconds and 62 seconds long.
        """
        rectified_signal = np.abs(self.signal)
        filtered_signal = self.butterworth_filter(rectified_signal)
        self.filtered_signal = filtered_signal

        # Detecting events based on threshold
        above_threshold = filtered_signal > self.threshold_value
        transitions = np.diff(above_threshold.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1

        # Handling edge cases
        if above_threshold[0]:
            starts = np.insert(starts, 0, 0)
        if above_threshold[-1]:
            ends = np.append(ends, len(filtered_signal) - 1)

        # Filter events to only include those between 55 and 62 seconds long
        min_duration_samples = int(52 * self.sample_rate)
        max_duration_samples = int(65 * self.sample_rate)
        valid_events = ((ends - starts) >= min_duration_samples) & ((ends - starts) <= max_duration_samples)
        starts = starts[valid_events]
        ends = ends[valid_events]

        self.df_triggers = pd.DataFrame({
            'start_index': starts,
            'end_index': ends,
            'duration_samples': ends - starts,
            'start_time (s)': starts / self.sample_rate,
            'end_time (s)': ends / self.sample_rate,
            'duration_time (s)': (ends - starts) / self.sample_rate
        })

    def plot_triggers(self):
        """
        Plots the filtered signal with detected trigger periods highlighted.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.butterworth_filter(np.abs(self.signal)), label='Filtered Signal')
        for start, end in zip(self.df_triggers['start_index'], self.df_triggers['end_index']):
            plt.axvspan(start, end, color='red', alpha=0.3)
        plt.title('Filtered Signal with Detected Trigger Periods')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def save_triggers(self):
        """
        Saves the DataFrame of detected triggers to a CSV file.

        :param filepath: str, the path to save the CSV file
        """
        self.df_triggers.to_csv(self.loader.folder_path + f'/{self.loader.name}' + '/triggers.csv')

    def plot_windows(self):
        images_dir = self.loader.folder_path + f'/{self.loader.name}' + '/window plots'
        os.makedirs(images_dir, exist_ok=True)

        # Loop through each pair of consecutive triggers
        for i in range(len(self.df_triggers) - 1):
            segment = self.filtered_signal[
                      self.df_triggers.loc[i, 'end_index']:self.df_triggers.loc[i + 1, 'start_index']]
            start_time_sec = self.df_triggers.loc[i, 'end_time (s)'] / 60
            end_time_sec = self.df_triggers.loc[i + 1, 'start_time (s)'] / 60
            time_vector = np.linspace(start_time_sec, end_time_sec, num=len(segment), endpoint=False)

            plt.figure(figsize=(12, 6))
            plt.plot(time_vector, segment)
            plt.title(f'Signal Segment Between {int(start_time_sec)} and {int(end_time_sec)} minutes')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Amplitude')
            plt.ylim(0, 300)
            plt.savefig(f'{images_dir}/plot_{i}.png')
            plt.close()

    def filter_bad_windows(self, clf_path=None, classes_path=None):
        """
        Uses ResNet-50 + log-reg pipeline to drop triggers whose
        adjoining window plots are classified as 'bad'. It then rewrites
        triggers.csv in the same folder.

        Call *after* plot_windows(), e.g.:
            td.plot_windows()
            td.filter_bad_windows()  # Uses built-in models automatically
        
        :param clf_path: str, optional path to custom classifier (.pkl file)
        :param classes_path: str, optional path to custom classes (.npy file)
        """
        
        # Use built-in models if no custom paths provided
        if clf_path is None or classes_path is None:
            import pkg_resources
            try:
                clf_path = pkg_resources.resource_filename('krembil_kit', 'logreg_good_bad.pkl')
                classes_path = pkg_resources.resource_filename('krembil_kit', 'good_bad_classes.npy')
                print("Using built-in ML models for window filtering")
            except Exception as e:
                raise FileNotFoundError(f"Built-in ML models not found. Please provide clf_path and classes_path parameters. Error: {e}")

        # --- set up models ---------------------------------------------------
        transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        clf = joblib.load(clf_path)
        class_names = np.load(classes_path)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        resnet = resnet50(pretrained=True)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

        # helper ----------------------------------------------------------------
        def predict_png(path):
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                feats = resnet(transform(img).unsqueeze(0).to(device))
                feats = feats.view(feats.size(0), -1).cpu().numpy()
            return class_names[clf.predict(feats)[0]]

        # --- walk through window-plot images ----------------------------------
        img_dir = os.path.join(self.loader.folder_path,
                               f"{self.loader.name}", "window plots")
        pngs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")],
                      key=lambda f: int(re.findall(r"\d+", f)[0]))

        bad_trigger_rows = set()  # indices to drop later

        for fname in pngs:
            win_idx = int(re.findall(r"\d+", fname)[0])  # n in plot_n.png
            prediction = predict_png(os.path.join(img_dir, fname))
            if prediction.lower() == "bad":
                # window between trigger n and n+1  → remove the *right* trigger
                bad_trigger_rows.add(win_idx + 1)

        # --- drop rows, re-index, save ----------------------------------------
        if bad_trigger_rows:
            self.df_triggers = (self.df_triggers
                                .drop(self.df_triggers.index[list(bad_trigger_rows)])
                                .reset_index(drop=True))

        # overwrite CSV
        save_path = os.path.join(self.loader.folder_path,
                                 f"{self.loader.name}", "triggers.csv")
        self.df_triggers.to_csv(save_path, index=False)
        print(f"Filtered triggers saved to {save_path}")

    def convert_to_video(self):
        def sort_key(filename):
            # Extract the number from the filename using regular expression
            numbers = re.findall(r'\d+', filename)
            return int(numbers[0]) if numbers else 0

        folder_path = self.loader.folder_path + f'/{self.loader.name}' + '/window plots'
        output_path = self.loader.folder_path + f'/{self.loader.name}' + '/trigger.mp4'

        # List files, sort them using the sort_key, and ensure they are .png files
        files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        sorted_files = sorted(files, key=sort_key)

        # Initialize OpenCV video writer
        frame = cv2.imread(os.path.join(folder_path, sorted_files[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used
        video = cv2.VideoWriter(output_path, fourcc, 10, (width, height))  # Increase fps here

        # Read each file, sort them numerically, and write to the video
        for filename in sorted_files:
            img = cv2.imread(os.path.join(folder_path, filename))
            video.write(img)

        video.release()  # Release the video writer
        cv2.destroyAllWindows()  # Close all OpenCV windows