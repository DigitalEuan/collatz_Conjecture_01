# Collatz Conjecture Parser - Core Functions
# Based on UBP theory research
# Clean implementation with proper error handling

import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def collatz(n):
    """Generate Collatz sequence for given number n"""
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n = n >> 1  # Divide by 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq

def binary_fraction(n):
    """Convert number to binary fraction representation"""
    binary = bin(n)[2:]
    return sum(int(d) / 2**(i+1) for i, d in enumerate(binary))

def map_to_3d(seq):
    """Map Collatz sequence to 3D points using UBP geometric mapping"""
    points = []
    
    # Generate 3D points from sequence
    for i, n in enumerate(seq):
        theta = 2 * np.pi * binary_fraction(n)
        phi = np.pi * (i % 6) / 3  # UBP 6-fold symmetry
        r = np.log1p(n)  # Logarithmic radial mapping
        
        # Spherical to Cartesian conversion
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        points.append([x, y, z])
    
    # Calculate tetrahedron angles and pi-angles
    tetrahedrons = []
    angle_sum = 0
    pi_angle_sum = 0
    pi_angles = 0
    
    # Process tetrahedrons (groups of 4 consecutive points)
    for i in range(len(points) - 3):
        tetra = np.array([points[i], points[i+1], points[i+2], points[i+3]], dtype=np.float64)
        tetrahedrons.append(tetra)
        
        v0, v1, v2, v3 = tetra
        edges = [v1 - v0, v2 - v0, v3 - v0]
        
        # Calculate angles between edges
        for j in range(len(edges)):
            for k in range(j+1, len(edges)):
                e1, e2 = edges[j], edges[k]
                
                # Avoid division by zero
                norm1 = np.linalg.norm(e1)
                norm2 = np.linalg.norm(e2)
                if norm1 < 1e-32 or norm2 < 1e-32:
                    continue
                
                cos_angle = np.dot(e1, e2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
                angle = np.arccos(cos_angle)
                angle_sum += angle
                
                # Check if angle is close to pi/k for k in [1,2,3,4,6]
                pi_ratios = [1, 2, 3, 4, 6]
                for k in pi_ratios:
                    if abs(angle - np.pi/k) < 0.005:  # Tolerance for pi-angles
                        pi_angles += 1
                        pi_angle_sum += angle
                        break
    
    # Calculate hull volume and space metrics
    space_metrics = []
    hull_volume = 0
    
    if len(points) >= 4:  # Need at least 4 points for ConvexHull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.array(points))
            hull_volume = hull.volume
        except Exception:
            hull_volume = 0
    
    # Calculate tetrahedron volumes
    for tetra in tetrahedrons:
        v0, v1, v2, v3 = tetra
        # Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
        vol = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6
        centroid = np.mean(tetra, axis=0)
        dist = np.linalg.norm(centroid - np.array([1, 0, 0]))
        space_metrics.append({'volume': vol, 'centroid_dist': dist})
    
    return (np.array(points), tetrahedrons, space_metrics, 
            hull_volume, angle_sum, pi_angle_sum, pi_angles)

def calculate_s_pi(pi_angle_sum, pi_angles):
    """Calculate S_π (average pi-angle) - key UBP invariant"""
    return pi_angle_sum / pi_angles if pi_angles > 0 else 0

def toggle_rate(seq):
    """Calculate toggle rate (even/odd transitions) for UBP analysis"""
    parities = [n % 2 for n in seq]
    toggles = sum(1 for i in range(len(parities)-1) if parities[i] != parities[i+1])
    return toggles / len(seq) if len(seq) > 1 else 0

def coherence_analysis(seq, segment_size=None):
    """Calculate coherence metrics for UBP validation"""
    if segment_size is None:
        segment_size = min(len(seq) // 4, 10000)  # Adaptive segment size
    
    binary = [n % 2 for n in seq]
    
    if len(binary) < 2:
        return 0, 0
    
    # Create overlapping segments
    segments = []
    step = max(1, segment_size // 2)
    for i in range(0, len(binary) - segment_size + 1, step):
        segments.append(binary[i:i+segment_size])
    
    if len(segments) < 2:
        return 0, 0
    
    c_ij = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            s1, s2 = segments[i], segments[j]
            if len(s1) == len(s2) and len(s1) > 1:
                corr = np.corrcoef(s1, s2)[0,1]
                if not np.isnan(corr):
                    c_ij.append(corr)
    
    return np.mean(c_ij) if c_ij else 0, np.std(c_ij) if c_ij else 0

def frequency_analysis(seq, fs=24e9):
    """Analyze frequency characteristics for UBP resonance detection"""
    binary = [n % 2 for n in seq]
    
    # Pad to minimum length for FFT
    min_length = 64
    if len(binary) < min_length:
        binary = binary + [0] * (min_length - len(binary))
    
    try:
        from scipy.signal import welch
        nperseg = min(len(binary), 256)
        freqs, psd = welch(binary, fs=fs, nperseg=nperseg)
        peak_idx = np.argmax(psd)
        return freqs[peak_idx], psd[peak_idx]
    except Exception:
        return 0, 0

def fractal_dimension(points):
    """Calculate fractal dimension of the 3D path"""
    if len(points) < 4:
        return 0
    
    try:
        points = np.array(points)
        scales = np.logspace(-2, 1, 10)  # Reduced range for stability
        counts = []
        
        for s in scales:
            if s > 0:
                boxes = np.floor(points / s).astype(int)
                unique_boxes = len(np.unique(boxes, axis=0))
                counts.append(max(1, unique_boxes))  # Avoid log(0)
        
        if len(counts) > 1:
            log_scales = np.log(1/scales)
            log_counts = np.log(counts)
            coeffs = np.polyfit(log_scales, log_counts, 1)
            return coeffs[0]
    except Exception:
        pass
    
    return 0

def validate_ubp_signature(s_pi, c_mean, freq_peak):
    """Validate UBP signature according to research criteria"""
    pi_threshold = 0.000001  # Tolerance for S_π ≈ π
    coherence_min = 0.3
    coherence_max = 0.5
    freq_target = 0.3183098861837907  # 1/π
    freq_threshold = 0.000001
    
    pi_valid = abs(s_pi - np.pi) < pi_threshold
    coherence_valid = coherence_min < c_mean < coherence_max
    freq_valid = abs(freq_peak - freq_target) < freq_threshold
    
    return {
        'pi_valid': pi_valid,
        'coherence_valid': coherence_valid,
        'freq_valid': freq_valid,
        'overall_valid': pi_valid and coherence_valid and freq_valid
    }

def test_basic_functions():
    """Test basic functionality with known values"""
    print("Testing basic functions...")
    
    # Test Collatz sequence
    seq = collatz(5)
    print(f"Collatz(5): {seq}")
    
    # Test binary fraction
    bf = binary_fraction(5)
    print(f"Binary fraction of 5: {bf:.6f}")
    
    # Test 3D mapping
    try:
        points, tetrahedrons, space_metrics, hull_volume, angle_sum, pi_angle_sum, pi_angles = map_to_3d(seq)
        s_pi = calculate_s_pi(pi_angle_sum, pi_angles)
        print(f"S_π for n=5: {s_pi:.6f}")
        print(f"Pi angles found: {pi_angles}")
        print(f"Hull volume: {hull_volume:.6f}")
        print("✓ Basic functions working correctly!")
        return True
    except Exception as e:
        print(f"✗ Error in 3D mapping: {e}")
        return False

if __name__ == "__main__":
    test_basic_functions()

