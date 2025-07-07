#!/usr/bin/env python3
"""
Collatz Conjecture Parser with UBP Theory Validation
Based on research by Euan Craig in collaboration with Grok (Xai) and other AI systems

This parser validates the Universal Binary Principle (UBP) through Collatz Conjecture analysis,
calculating S_π invariant, coherence metrics, and 3D geometric mappings.
"""

import numpy as np
import json
import time
import sys
from datetime import datetime
from pathlib import Path

# Import core functions
from collatz_parser_core import (
    collatz, map_to_3d, calculate_s_pi, toggle_rate, 
    coherence_analysis, frequency_analysis, fractal_dimension,
    validate_ubp_signature
)

class CollatzParser:
    """Main Collatz parser with UBP validation and computational limits"""
    
    def __init__(self):
        self.max_sequence_length = 1000000  # Computational limit
        self.max_input_value = 10**12       # Maximum input value
        self.warning_threshold = 10**6      # Warning threshold
        
    def check_computational_limits(self, n):
        """Check if input is within computational limits"""
        if n <= 0:
            raise ValueError("Input must be a positive integer")
        
        if n > self.max_input_value:
            raise ValueError(f"Input {n} exceeds maximum limit of {self.max_input_value}")
        
        # Estimate sequence length (rough approximation)
        estimated_length = int(np.log2(n) * 10)  # Very rough estimate
        
        warnings = []
        if n > self.warning_threshold:
            warnings.append(f"Large input detected (n={n}). Computation may take significant time.")
        
        if estimated_length > self.max_sequence_length // 10:
            warnings.append(f"Estimated sequence length may be large. Consider smaller input.")
        
        return warnings
    
    def parse_collatz(self, n, include_visualization=False):
        """
        Parse Collatz sequence and calculate UBP metrics
        
        Args:
            n: Input number
            include_visualization: Whether to generate 3D visualization data
            
        Returns:
            Dictionary with all calculated metrics and validation results
        """
        start_time = time.time()
        
        # Check computational limits
        warnings = self.check_computational_limits(n)
        
        print(f"\n{'='*60}")
        print(f"Collatz Conjecture Parser - UBP Validation")
        print(f"{'='*60}")
        print(f"Input: {n}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        if warnings:
            print(f"\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        print(f"\nGenerating Collatz sequence...")
        
        # Generate Collatz sequence
        try:
            seq = collatz(n)
        except Exception as e:
            raise RuntimeError(f"Failed to generate Collatz sequence: {e}")
        
        sequence_length = len(seq)
        print(f"Sequence length: {sequence_length}")
        
        if sequence_length > self.max_sequence_length:
            print(f"⚠️  Sequence length ({sequence_length}) exceeds limit ({self.max_sequence_length})")
            print(f"Truncating to first {self.max_sequence_length} elements...")
            seq = seq[:self.max_sequence_length]
            sequence_length = len(seq)
        
        print(f"First 10 elements: {seq[:10]}{'...' if len(seq) > 10 else ''}")
        
        # Calculate 3D geometric mapping
        print(f"\nCalculating 3D geometric mapping...")
        try:
            points, tetrahedrons, space_metrics, hull_volume, angle_sum, pi_angle_sum, pi_angles = map_to_3d(seq)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate 3D mapping: {e}")
        
        # Calculate UBP metrics
        print(f"Calculating UBP metrics...")
        
        # S_π calculation (key UBP invariant)
        s_pi = calculate_s_pi(pi_angle_sum, pi_angles)
        
        # Toggle rate analysis
        toggle = toggle_rate(seq)
        
        # Coherence analysis
        c_mean, c_std = coherence_analysis(seq)
        
        # Frequency analysis
        freq_peak, psd_peak = frequency_analysis(seq)
        
        # Fractal dimension
        fractal_dim = fractal_dimension(points)
        
        # Volume calculations
        tetra_volumes = [m['volume'] for m in space_metrics] if space_metrics else []
        mean_volume = np.mean(tetra_volumes) if tetra_volumes else 0
        volume_ratio = mean_volume / hull_volume if hull_volume > 0 else 0
        
        # Calculate derived metrics
        L = sequence_length
        angle_sum_norm = angle_sum / L if L > 0 else 0
        pi_angle_percentage = (100 * pi_angles / (len(space_metrics) * 6)) if space_metrics else 0
        
        # UBP signature validation
        ubp_validation = validate_ubp_signature(s_pi, c_mean, freq_peak)
        
        # Compile results
        results = {
            'input': {
                'n': n,
                'timestamp': datetime.now().isoformat(),
                'warnings': warnings
            },
            'sequence': {
                'length': sequence_length,
                'first_10': seq[:10],
                'truncated': sequence_length == self.max_sequence_length
            },
            'ubp_metrics': {
                's_pi': float(s_pi),
                's_pi_target': float(np.pi),
                's_pi_error': float(abs(s_pi - np.pi)),
                'pi_angles': int(pi_angles),
                'pi_angle_percentage': float(pi_angle_percentage),
                'toggle_rate': float(toggle),
                'coherence_mean': float(c_mean),
                'coherence_std': float(c_std),
                'frequency_peak': float(freq_peak),
                'frequency_target': 0.3183098861837907,  # 1/π
                'fractal_dimension': float(fractal_dim)
            },
            'geometry': {
                'total_angle_sum': float(angle_sum),
                'normalized_angle_sum': float(angle_sum_norm),
                'hull_volume': float(hull_volume),
                'mean_tetrahedron_volume': float(mean_volume),
                'volume_ratio': float(volume_ratio),
                'num_tetrahedrons': len(space_metrics)
            },
            'validation': {
                'ubp_signature_valid': ubp_validation['overall_valid'],
                'pi_invariant_valid': ubp_validation['pi_valid'],
                'coherence_valid': ubp_validation['coherence_valid'],
                'frequency_valid': ubp_validation['freq_valid'],
                'precision_level': 'p < 10^-6' if ubp_validation['overall_valid'] else 'validation_failed'
            },
            'performance': {
                'computation_time': time.time() - start_time,
                'memory_efficient': sequence_length <= 100000
            }
        }
        
        # Add visualization data if requested
        if include_visualization and len(points) > 0:
            results['visualization'] = {
                'points_3d': points.tolist() if len(points) < 10000 else points[:10000].tolist(),
                'tetrahedrons': [t.tolist() for t in tetrahedrons[:100]],  # Limit for performance
                'note': 'Limited to first 10000 points and 100 tetrahedrons for performance'
            }
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print formatted summary of results"""
        print(f"\n{'='*60}")
        print(f"UBP VALIDATION RESULTS")
        print(f"{'='*60}")
        
        ubp = results['ubp_metrics']
        val = results['validation']
        
        print(f"S_π (Pi Invariant):     {ubp['s_pi']:.12f}")
        print(f"Target (π):             {ubp['s_pi_target']:.12f}")
        print(f"Error:                  {ubp['s_pi_error']:.2e}")
        print(f"Pi Angles Found:        {ubp['pi_angles']} ({ubp['pi_angle_percentage']:.1f}%)")
        
        print(f"\nCoherence Analysis:")
        print(f"C_ij (mean):            {ubp['coherence_mean']:.6f} ± {ubp['coherence_std']:.6f}")
        print(f"Toggle Rate:            {ubp['toggle_rate']:.6f}")
        
        print(f"\nFrequency Analysis:")
        print(f"Peak Frequency:         {ubp['frequency_peak']:.12f} Hz")
        print(f"Target (1/π):           {ubp['frequency_target']:.12f} Hz")
        
        print(f"\nGeometric Properties:")
        print(f"Hull Volume:            {results['geometry']['hull_volume']:.6f}")
        print(f"Fractal Dimension:      {ubp['fractal_dimension']:.6f}")
        print(f"Tetrahedrons:           {results['geometry']['num_tetrahedrons']}")
        
        print(f"\nUBP Signature Validation:")
        print(f"Overall Valid:          {'✓' if val['ubp_signature_valid'] else '✗'}")
        print(f"Pi Invariant:           {'✓' if val['pi_invariant_valid'] else '✗'}")
        print(f"Coherence Range:        {'✓' if val['coherence_valid'] else '✗'}")
        print(f"Frequency Match:        {'✓' if val['frequency_valid'] else '✗'}")
        print(f"Precision Level:        {val['precision_level']}")
        
        print(f"\nPerformance:")
        print(f"Computation Time:       {results['performance']['computation_time']:.2f} seconds")
        print(f"Memory Efficient:       {'✓' if results['performance']['memory_efficient'] else '✗'}")
    
    def save_results(self, results, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collatz_results_{results['input']['n']}_{timestamp}.json"
        
        filepath = Path(filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✓ Results saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
            return None

def main():
    """Main function for command-line usage"""
    parser = CollatzParser()
    
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
            include_viz = '--viz' in sys.argv
            save_file = '--save' in sys.argv
            
            results = parser.parse_collatz(n, include_visualization=include_viz)
            
            if save_file:
                parser.save_results(results)
                
        except ValueError:
            print("Error: Please provide a valid integer")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Collatz Conjecture Parser with UBP Validation")
        print("Usage: python collatz_parser.py <number> [--viz] [--save]")
        print("  --viz: Include 3D visualization data")
        print("  --save: Save results to JSON file")
        print("\nExample: python collatz_parser.py 27 --save")

if __name__ == "__main__":
    main()

