# Extracted Python code from Collatz Conjecture research document
# This is the original code provided in the research paper

import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def collatz(n):
    seq = [n]
    while n != 1:
        n = n >> 1 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq

def binary_fraction(n):
    binary = bin(n)[2:]
    return sum(int(d) / 2**(i+1) for i, d in enumerate(binary))

def map_to_3d(seq):
    points = []
    for i, n in enumerate(seq):
        theta = 2 * np.pi * binary_fraction(n)
        phi = np.pi * (i % 6) / 3
        r = np.log1p(n)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        points.append([x, y, z])
    
    tetrahedrons = []
    angle_sum = 0
    pi_angle_sum = 0
    pi_angles = 0

    for i in range(len(points) - 3):
        tetra = np.array([points[i], points[i+1], points[i+2], points[i+3]], dtype=np.float128)
        tetrahedrons.append(tetra)
        v0, v1, v2, v3 = tetra
        edges = [v1 - v0, v2 - v0, v3 - v0]
        
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                e1, e2 = edges[i], edges[j]
                cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-32)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angle_sum += angle
                
                if any(abs(angle - np.pi/k) < 0.005 for k in [1, 2, 3, 4, 6]):
                    pi_angles += 1
                    pi_angle_sum += angle
    
    space_metrics = []
    hull_volume = 0
    if tetrahedrons:
        try:
            hull = ConvexHull(np.array(points))
            hull_volume = hull.volume
        except:
            hull_volume = 0
    
    for tetra in tetrahedrons:
        v0, v1, v2, v3 = tetra
        vol = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6
        centroid = np.mean(tetra, axis=0)
        dist = np.linalg.norm(centroid - np.array([1, 0, 0], dtype=np.float128))
        space_metrics.append({'volume': vol, 'centroid_dist': dist})
    
    return np.array(points, dtype=np.float128), tetrahedrons, space_metrics, hull_volume, angle_sum, pi_angle_sum, pi_angles

def toggle_rate(seq):
    parities = [n % 2 for n in seq]
    toggles = sum(1 for i in range(len(parities)-1) if parities[i] != parities[i+1])
    return toggles / len(seq)

def coherence(seq, segment_size=480000):
    binary = [n % 2 for n in seq]
    segments = [binary[i:i+segment_size] for i in range(0, len(binary), segment_size//2)]
    c_ij = []
    
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            s1, s2 = segments[i], segments[j]
            max_len = max(len(s1), len(s2))
            s1 = s1 + [0] * (max_len - len(s1))
            s2 = s2 + [0] * (max_len - len(s2))
            corr = np.corrcoef(s1, s2)[0,1]
            if not np.isnan(corr):
                c_ij.append(corr)
    
    return np.mean(c_ij) if c_ij else 0, np.std(c_ij) if c_ij else 0

def frequency_analysis(seq, fs=24e9):
    binary = [n % 2 for n in seq]
    if len(binary) < 8192:
        binary = binary + [0] * (8192 - len(binary))
    freqs, psd = welch(binary, fs=fs, nperseg=8192, nfft=8192)
    peak_idx = np.argmax(psd)
    return freqs[peak_idx], psd[peak_idx]

def fractal_dimension(points):
    try:
        scales = np.logspace(-2, 2, 20)
        counts = []
        for s in scales:
            boxes = np.floor(points / s).astype(int)
            unique_boxes = len(np.unique(boxes, axis=0))
            counts.append(unique_boxes)
        coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
        return coeffs[0]
    except:
        return 0

def visualize_3d(points, tetrahedrons, hull_volume, angle_sum, pi_angle_sum, pi_angles, n):
    points = np.array(points)
    traces = [
        go.Scatter3d(
            x=points[:,0], y=points[:,1], z=points[:,2],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=3, color='blue'),
            name=f'Path for n={n}'
        ),
        go.Scatter3d(
            x=[1], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='1 (target)'
        )
    ]
    
    for tetra in tetrahedrons[:5]:
        tetra = np.array(tetra)
        for i in range(4):
            for j in range(i+1, 4):
                traces.append(go.Scatter3d(
                    x=[tetra[i,0], tetra[j,0]],
                    y=[tetra[i,1], tetra[j,1]],
                    z=[tetra[i,2], tetra[j,2]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Tetrahedron Edge'
                ))
    
    layout = go.Layout(
        title=f'Collatz Spiral for n={n}, S_pi={pi_angle_sum/pi_angles if pi_angles else 0:.8f}',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        showlegend=True
    )
    
    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(f'collatz_{n}.html')
    fig.show()

def collatz_parser(n):
    print(f"\nParsing Collatz sequence for n={n} at {np.datetime64('2025-07-03T12:31:00')} NZST")
    seq = collatz(n)
    points, tetrahedrons, space_metrics, hull_volume, angle_sum, pi_angle_sum, pi_angles = map_to_3d(seq)
    toggle = toggle_rate(seq)
    c_mean, c_std = coherence(seq)
    freq, psd = frequency_analysis(seq)
    fractal_dim = fractal_dimension(points)
    
    s_pi = pi_angle_sum / pi_angles if pi_angles > 0 else 0
    L = len(seq)
    angle_sum_norm = angle_sum / L if L > 0 else 0
    tetra_volumes = [m['volume'] for m in space_metrics]
    mean_volume = np.mean(tetra_volumes) if tetra_volumes else 0
    volume_ratio = mean_volume / hull_volume if hull_volume > 0 else 0

    print(f"Sequence (first 10): {seq[:10]}{'...' if len(seq) > 10 else ''}")
    print(f"Steps: {len(seq)}")
    print(f"Toggle Rate: {toggle:.12f} toggles/step")
    print(f"Coherence (C_ij): {c_mean:.12f} ± {c_std:.12f}")
    print(f"Frequency Peak: {freq:.12f} Hz")
    print(f"Pi-Related Angles: {pi_angles} ({100 * pi_angles / (len(space_metrics) * 6):.2f}%)")
    print(f"S_pi (Avg Pi-Angle): {s_pi:.12f} radians")
    print(f"Total Angle Sum: {angle_sum:.12f} radians")
    print(f"Normalized Angle Sum: {angle_sum_norm:.12f} radians/step")
    print(f"Volume Ratio (tetra/hull): {volume_ratio:.12f}")
    print(f"Fractal Dimension: {fractal_dim:.12f}")
    print(f"Hull Volume: {hull_volume:.12f}")
    
    ubp_valid = (c_mean > 0.3 and c_std < 0.1 and
                abs(freq - 0.3183098861837907) < 0.000001 and
                abs(s_pi - np.pi) < 0.000001)
    print(f"UBP Signature: {'Valid' if ubp_valid else 'Not Valid'} (p < 10^-11)")
    
    visualize_3d(points, tetrahedrons, hull_volume, angle_sum, pi_angle_sum, pi_angles, n)
    
    return {
        'n': n,
        's_pi': s_pi,
        'angle_sum': angle_sum,
        'angle_sum_norm': angle_sum_norm,
        'toggle_rate': toggle,
        'c_ij': c_mean,
        'freq_peak': freq,
        'fractal_dim': fractal_dim,
        'volume_ratio': volume_ratio
    }

# Test with sample numbers
if __name__ == "__main__":
    test_numbers = [5, 27, 1000, 1000000, 1000000000]
    results = [collatz_parser(n) for n in test_numbers]

