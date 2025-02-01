import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def read_s2p(file_path):
    """Reads S2P file and returns frequencies, Gain (dB), and Phase (Degrees)"""
    frequencies, gain_db, phase_deg = [], [], []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('!') or line.startswith('#'):
                continue  # Skip comment lines
            values = line.strip().split()
            if len(values) < 5:
                continue  # Skip invalid lines
            try:
                freq = float(values[0])  # Frequency
                gain_db.append(float(values[3]))  # Gain (dB) directly from the file
                phase_deg.append(float(values[4]))  # Phase (Degrees) directly from the file
                frequencies.append(freq)
            except ValueError:
                continue  # Skip invalid data

    return np.array(frequencies), np.array(gain_db), np.array(phase_deg)


def extract_values_at_freq(file_paths, target_freq_ghz):
    """Extract Gain and Phase at a specific frequency from all files"""
    target_freq_hz = target_freq_ghz * 1e9  # Convert to Hz
    gain_values, phase_values = [], []

    for file_path in file_paths:
        freqs, gains, phases = read_s2p(file_path)
        if len(freqs) == 0:
            continue  # Skip empty files
        idx = (np.abs(freqs - target_freq_hz)).argmin()  # Find index of closest frequency
        gain_values.append(gains[idx])
        phase_values.append(phases[idx])

    return np.array(gain_values), np.array(phase_values)


def compute_cdf(data, num_points=18):
    """Computes a smooth CDF using Kernel Density Estimation without scipy"""
    # Step 1: Perform KDE on the data to estimate the PDF
    x_smooth, pdf_smooth = smooth_kernel_density(data, num_points=num_points)

    # Step 2: Compute the CDF by numerically integrating the PDF
    cdf_smooth = np.cumsum(pdf_smooth) * (x_smooth[1] - x_smooth[0])  # Approximate integral (Riemann sum)

    # Step 3: Normalize the CDF to be between 0 and 1
    cdf_smooth /= cdf_smooth[-1]  # Normalize the CDF to the range [0, 1]

    return x_smooth, cdf_smooth


def smooth_kernel_density(data, bandwidth=None, num_points=18):
    """Perform Kernel Density Estimation without scipy using Gaussian smoothing"""
    min_data, max_data = min(data), max(data)
    x_values = np.linspace(min_data, max_data, num_points)
    kde_values = np.zeros_like(x_values)

    if bandwidth is None:
        bandwidth = 1.06 * np.std(data) * len(data) ** (-1 / 5)

    # Apply Gaussian kernel for each data point
    for d in data:
        kde_values += np.exp(-0.5 * ((x_values - d) / bandwidth) ** 2)

    # Normalize the result to form a proper density
    kde_values /= np.sum(kde_values)
    return x_values, kde_values


def plot_pdf_cdf(gain_values, phase_values, target_freq_ghz):
    """Displays PDF and CDF plots for Gain and Phase"""

    # Perform Kernel Density Estimation for Gain and Phase
    gain_x, gain_kde = smooth_kernel_density(gain_values)
    phase_x, phase_kde = smooth_kernel_density(phase_values)

    # Compute CDF for Gain and Phase
    gain_cdf_x, gain_cdf_y = compute_cdf(gain_values)
    phase_cdf_x, phase_cdf_y = compute_cdf(phase_values)

    # Ensure CDF is non-decreasing (monotonic)
    gain_cdf_y = np.clip(gain_cdf_y, 0, 1)
    phase_cdf_y = np.clip(phase_cdf_y, 0, 1)

    # Find the 50% point for vertical lines
    gain_50_percent = np.percentile(gain_values, 50)
    phase_50_percent = np.percentile(phase_values, 50)

    # Find the maximum KDE points for Gain and Phase
    gain_max_idx = np.argmax(gain_kde)
    phase_max_idx = np.argmax(phase_kde)

    # Create a subplot layout with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f"64 Channels Gain PDF at {target_freq_ghz} GHz", f"64 Channels Gain CDF at {target_freq_ghz} GHz",
                        f"64 Channels Phase PDF at {target_freq_ghz} GHz", f"64 Channels Phase CDF at {target_freq_ghz} GHz"),
        column_widths=[0.5, 0.5], row_heights=[0.5, 0.5],
        shared_xaxes=False, shared_yaxes=False
    )

    # Add Gain PDF (smoothed curve) as line plot
    fig.add_trace(go.Scatter(x=gain_x, y=gain_kde, mode='lines', name=f'Gain PDF - 50%: {gain_50_percent:.2f} dB',
                             line=dict(color='#1f77b4', width=5)),
                  row=1, col=1)

    # Add the maximum point for Gain PDF
    fig.add_trace(go.Scatter(x=[gain_x[gain_max_idx]], y=[gain_kde[gain_max_idx]], mode='markers',
                             name=f'Max Gain Point: {gain_x[gain_max_idx]:.2f} dB', marker=dict(color='red', size=8)),
                  row=1, col=1)

    # Add Gain CDF plot
    fig.add_trace(
        go.Scatter(x=gain_cdf_x, y=gain_cdf_y, mode='lines', name=f'Gain CDF - 50%: {gain_50_percent:.2f} dB',
                   line=dict(color='#17becf', width=5)),
        row=1, col=2)

    # Add Phase PDF (smoothed curve) as line plot
    fig.add_trace(go.Scatter(x=phase_x, y=phase_kde, mode='lines', name=f'Phase PDF - 50%: {phase_50_percent:.2f}°',
                             line=dict(color='#ff7f0e', width=5)),
                  row=2, col=1)

    # Add the maximum point for Phase PDF
    fig.add_trace(go.Scatter(x=[phase_x[phase_max_idx]], y=[phase_kde[phase_max_idx]], mode='markers',
                             name=f'Max Phase Point: {phase_x[phase_max_idx]:.2f}°', marker=dict(color='red', size=8)),
                  row=2, col=1)

    # Add Phase CDF plot
    fig.add_trace(
        go.Scatter(x=phase_cdf_x, y=phase_cdf_y, mode='lines', name=f'Phase CDF - 50%: {phase_50_percent:.2f}°',
                   line=dict(color='#d62728', width=5)),
        row=2, col=2)

    # Add 50% vertical line for Gain PDF and CDF
    fig.add_trace(go.Scatter(x=[gain_50_percent, gain_50_percent], y=[0, max(gain_kde)], mode='lines',
                             line=dict(dash='dash', color='gray'), name='50% Line'),
                  row=1, col=1)

    fig.add_trace(
        go.Scatter(x=[gain_50_percent, gain_50_percent], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'),
                   name='50% Line'),
        row=1, col=2)

    # Add 50% vertical line for Phase PDF and CDF
    fig.add_trace(go.Scatter(x=[phase_50_percent, phase_50_percent], y=[0, max(phase_kde)], mode='lines',
                             line=dict(dash='dash', color='gray'), name='50% Line'),
                  row=2, col=1)

    fig.add_trace(
        go.Scatter(x=[phase_50_percent, phase_50_percent], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'),
                   name='50% Line'),
        row=2, col=2)

    # Update layout for all subplots
    fig.update_layout(
        title=f"Tx Mode PDF and CDF at {target_freq_ghz} GHz",
        plot_bgcolor='rgb(243, 243, 243)',
        template='plotly_dark',
        font=dict(family="Arial, sans-serif", size=14, color='white'),
        showlegend=True,
        height=920,
        width=1900,
        autosize=True
    )

    # Set axis labels
    fig.update_xaxes(title_text='Gain (dB)', row=1, col=1)
    fig.update_yaxes(title_text='Density Probability', row=1, col=1)
    fig.update_xaxes(title_text='Gain (dB)', row=1, col=2)
    fig.update_yaxes(title_text='Cumulative Probability', row=1, col=2)
    fig.update_xaxes(title_text='Phase (Degrees)', row=2, col=1)
    fig.update_yaxes(title_text='Density Probability', row=2, col=1)
    fig.update_xaxes(title_text='Phase (Degrees)', row=2, col=2)
    fig.update_yaxes(title_text='Cumulative Probability', row=2, col=2)

    # Show figure and save it to an HTML file
    fig.write_html("output_plots.html", include_plotlyjs=True)
    fig.show()


# Running the software
if __name__ == "__main__":
    # Get all files from a directory
    file_directory = input("Enter the directory containing the S2P files: ").strip()
    if not os.path.isdir(file_directory):
        print("Invalid directory. Exiting the program.")
        exit()

    file_paths = [os.path.join(file_directory, f) for f in os.listdir(file_directory) if f.endswith('.s2p')]

    if not file_paths:
        print("No S2P files found in the directory. Exiting the program.")
        exit()

    # Get the frequency to check
    try:
        target_freq_ghz = float(input("Enter the frequency to check in GHz: "))
    except ValueError:
        print("Invalid frequency. Exiting the program.")
        exit()

    # Extract Gain and Phase from the files
    gain_values, phase_values = extract_values_at_freq(file_paths, target_freq_ghz)

    if len(gain_values) == 0 or len(phase_values) == 0:
        print("No data found for the specified frequency.")
        exit()

    # Display the plots
    plot_pdf_cdf(gain_values, phase_values, target_freq_ghz)
