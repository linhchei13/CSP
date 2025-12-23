"""
2D Bin Packing Problem - Non-Stacking Version with MIP using OR-Tools
With Rotation and Symmetry Breaking constraints
"""

from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os
import sys
import signal
import math
import pandas as pd
import timeit
import subprocess
import traceback
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Global variables to track best solution found so far
best_bins = float('inf')
best_assignments = []
best_positions = []
best_rotations = []
upper_bound = 0

# Signal handler for graceful interruption
def handle_interrupt(signum, frame):
    print(f"\nReceived interrupt signal {signum}. Saving current best solution.")
    
    current_bins = best_bins if best_bins != float('inf') else upper_bound
    print(f"Best bins found before interrupt: {current_bins}")
    
    # Save result as JSON for the controller to pick up
    result = {
        'Instance': instances[instance_id],
        'Runtime': timeit.default_timer() - start,
        'N_Bins': current_bins,
        'Status': 'TIMEOUT',
        'Allow_Rotation': 'Yes'
    }
    
    with open(f'results_OR-TOOLS_MIP_R_SB_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

# Create output folder if it doesn't exist
if not os.path.exists('OR-TOOLS_MIP_R_SB'):
    os.makedirs('OR-TOOLS_MIP_R_SB')

def read_file_instance(instance_name):
    """Read instance file based on instance name"""
    s = ''
    
    # Determine file path based on instance name
    
    # For other instances, try different folders
    possible_paths = [
        f"inputs/{instance_name}.txt",
        f"inputs/BENG/{instance_name}.txt",
        f"inputs/WANG/{instance_name}",
        f"inputs/NGCUT/{instance_name}",
        f"inputs/CGCUT/{instance_name}", 
        f"inputs/HIFI1997_format/{instance_name}",
        f"inputs/CHL_format/{instance_name}.txt",
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    if filepath is None:
        raise FileNotFoundError(f"Could not find instance file for {instance_name}")

    try:
        with open(filepath, 'r') as f:
            s = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file not found: {filepath}")
    
    return s.splitlines()

instances = [
    "",
    # BENG instances (10 instances)
    "BENG01", "BENG02", "BENG03", "BENG04", "BENG05",
    "BENG06", "BENG07", "BENG08", "BENG09", "BENG10",
    # WANG instances (3 instances)
    "WANG1", "WANG2", "WANG3",
    # ngcut (12 instances)
    "ngcut1", "ngcut2", "ngcut3", "ngcut4", "ngcut5", "ngcut6",
    "ngcut7", "ngcut8", "ngcut9", "ngcut10", "ngcut11", "ngcut12",
    # cgcut (3 instances)
    "cgcut1", "cgcut2", "cgcut3",
    # Hifi
    "A1", "A2", "A3", "A4", "A5", "HH",
    # CHL
    "CHL1", "CHL2", "CHL3", "CHL4", "CHL5", "CHL6", "CHL7",
    "Hchl1", "Hchl2", "Hchl3s", "Hchl4s", "Hchl5s", "Hchl6s",
    "Hchl7s","Hchl8s", "Hchl9",
     ]

def first_fit_upper_bound(rectangles, W, H, allow_rotation=True):
    """First-fit heuristic to get upper bound with rotation option"""
    # Each bin is a list of placed rectangles: (x, y, w, h, rotated)
    bins = []
    
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                rect = (x, y, w, h)
                overlap = False
                for (px, py, pw, ph, _) in bin_rects:
                    if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
        return None
    
    for rect in rectangles:
        original_w, original_h = rect[0], rect[1]
        placed = False
        
        # Try both orientations if rotation is allowed
        orientations = [(original_w, original_h, False)]
        if allow_rotation and original_w != original_h:
            orientations.append((original_h, original_w, True))
        
        # Try to place in existing bins
        for w, h, rotated in orientations:
            if placed:
                break
                
            for bin_rects in bins:
                pos = fits(bin_rects, w, h, W, H)
                if pos is not None:
                    bin_rects.append((pos[0], pos[1], w, h, rotated))
                    placed = True
                    break
        
        # If not placed, create a new bin
        if not placed:
            # Try the best orientation for a new bin
            best_w, best_h, best_rotated = orientations[0]
            if allow_rotation and len(orientations) > 1:
                # Choose orientation that leaves more space
                if orientations[1][0] <= W and orientations[1][1] <= H:
                    best_w, best_h, best_rotated = orientations[1]
            
            if best_w <= W and best_h <= H:
                bins.append([(0, 0, best_w, best_h, best_rotated)])
            else:
                # Rectangle doesn't fit in any bin
                return float('inf')
    
    return len(bins)

def calculate_lower_bound(rectangles, W, H):
    """Calculate lower bound for number of bins needed"""
    total_area = sum(w * h for w, h in rectangles)
    bin_area = W * H
    return math.ceil(total_area / bin_area)

def save_checkpoint(instance_id, bins, status="IN_PROGRESS"):
    """Save checkpoint for current progress"""
    checkpoint = {
        "Instance": instances[instance_id],
        'Runtime': timeit.default_timer() - start,
        'N_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status,
        'Allow_Rotation': 'Yes'
    }
    
    with open(f'checkpoint_OR-TOOLS_MIP_R_SB_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

def display_solution(W, H, rectangles, positions, assignments, rotations, instance_name):
    """Display solution with one subplot per bin, showing rotations"""
    n_bins = len(set(assignments))
    n_rectangles = len(rectangles)
    
    # Determine layout of subplots
    ncols = min(n_bins, 3)
    nrows = math.ceil(n_bins / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    fig.suptitle(f'Solution for {instance_name} - {n_bins} bins (with rotation)', fontsize=16)
    
    # Handle different subplot configurations
    if n_bins == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Create bins structure
    bins = [[] for _ in range(n_bins)]
    for i in range(n_rectangles):
        bins[assignments[i]].append(i)
    
    # Draw rectangles for each bin
    for bin_idx, items in enumerate(bins):
        if bin_idx < len(axes):
            ax = axes[bin_idx]
            ax.set_title(f'Bin {bin_idx + 1}')
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect('equal')
            
            # Draw each rectangle in this bin
            for item_idx in items:
                original_width, original_height = rectangles[item_idx]
                
                # Apply rotation if needed
                if rotations[item_idx]:
                    width, height = original_height, original_width
                else:
                    width, height = original_width, original_height
                
                x, y = positions[item_idx]
                
                rect = plt.Rectangle((x, y), width, height, 
                                   edgecolor='black', 
                                   facecolor=plt.cm.Set3(item_idx % 12),
                                   alpha=0.7)
                ax.add_patch(rect)
                
                # Add item number and rotation info
                rot_info = 'R' if rotations[item_idx] else 'NR'
                ax.text(x + width/2, y + height/2, f'{item_idx + 1}\n{rot_info}', 
                       ha='center', va='center', fontweight='bold')
            
            # Set grid and ticks
            ax.set_xticks(range(0, W+1, max(1, W//10)))
            ax.set_yticks(range(0, H+1, max(1, H//10)))
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f'OR-TOOLS_MIP_R_SB/{instance_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def solve_bin_packing(W, H, rectangles, lower_bound, upper_bound, time_limit=890):
    """
    Solve 2D Bin Packing using OR-Tools MIP with rotation and symmetry breaking
    
    Args:
        W: Width of each bin
        H: Height of each bin
        rectangles: List of (width, height) tuples
        lower_bound: Lower bound on number of bins
        upper_bound: Upper bound on number of bins
        time_limit: Time limit in seconds
    
    Returns:
        Dictionary with solution or None if no solution found
    """
    global best_bins, best_assignments, best_positions, best_rotations
    
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None
    
    # Set time limit
    solver.set_time_limit(time_limit * 1000)  # Time limit in milliseconds
    
    n = len(rectangles)
    max_bins = min(n, upper_bound)  # No need for more bins than items
    
    print(f"Creating MIP model with {n} items and up to {max_bins} bins...")
    start_model_time = time.time()
    
    # Variables
    # 1. Position variables
    x = {}  # x[i,b] = x-position of item i in bin b
    y = {}  # y[i,b] = y-position of item i in bin b
    for i in range(n):
        for b in range(max_bins):
            x[i,b] = solver.NumVar(0, W, f'x_{i}_{b}')
            y[i,b] = solver.NumVar(0, H, f'y_{i}_{b}')
    
    # 2. Assignment variables
    z = {}  # z[i,b] = 1 if item i is assigned to bin b
    for i in range(n):
        for b in range(max_bins):
            z[i,b] = solver.BoolVar(f'z_{i}_{b}')
    
    # 3. Bin usage variables
    u = {}  # u[b] = 1 if bin b is used
    for b in range(max_bins):
        u[b] = solver.BoolVar(f'u_{b}')
    
    # 4. Rotation variables
    rotate = {}  # rotate[i] = 1 if item i is rotated
    for i in range(n):
        rotate[i] = solver.BoolVar(f'rotate_{i}')
    
    # 5. Auxiliary variables for non-overlap (following debug logic)
    left = {}   # left[i,j,b] = 1 if item i is to the left of item j in bin b
    right = {}  # right[i,j,b] = 1 if item i is to the right of item j in bin b
    below = {}  # below[i,j,b] = 1 if item i is below item j in bin b
    above = {}  # above[i,j,b] = 1 if item i is above item j in bin b
    
    # Declare auxiliary variables for all pairs (i, j, b) where i != j
    for i in range(n):
        for j in range(n):
            if i != j:
                for b in range(max_bins):
                    left[i,j,b] = solver.BoolVar(f'left_{i}_{j}_{b}')
                    right[i,j,b] = solver.BoolVar(f'right_{i}_{j}_{b}')
                    below[i,j,b] = solver.BoolVar(f'below_{i}_{j}_{b}')
                    above[i,j,b] = solver.BoolVar(f'above_{i}_{j}_{b}')
    
    # Constraints
    
    # 1. Each item must be placed in exactly one bin
    for i in range(n):
        solver.Add(sum(z[i,b] for b in range(max_bins)) == 1)
    
    # 2. Bin usage constraints - if an item is in bin b, bin b must be used
    for b in range(max_bins):
        for i in range(n):
            solver.Add(z[i,b] <= u[b])
    
    # 3. Symmetry Breaking: Bins are used in order
    for b in range(1, max_bins):
        solver.Add(u[b] <= u[b-1])
    
    # Find largest rectangle by area for additional symmetry breaking
    max_area_idx = 0
    max_area = rectangles[0][0] * rectangles[0][1]
    
    for i in range(1, n):
        area = rectangles[i][0] * rectangles[i][1]
        if area > max_area:
            max_area = area
            max_area_idx = i
    
    # 4. Additional symmetry breaking: Place largest rectangle in first bin
    if n > 1:
        solver.Add(z[max_area_idx, 0] == 1)
        
        # Domain reduction for largest rectangle
        w_max, h_max = rectangles[max_area_idx]
        # Considering possible rotation for domain reduction
        solver.Add(x[max_area_idx, 0] <= (W - min(w_max, h_max)) // 2)
    
    # 5. Bound constraints considering rotation
    for i in range(n):
        w, h = rectangles[i]
        for b in range(max_bins):
            # Width constraint considering rotation
            solver.Add(x[i,b] + w * (1 - rotate[i]) + h * rotate[i] <= W + (W + H) * (1 - z[i,b]))
            # Height constraint considering rotation
            solver.Add(y[i,b] + h * (1 - rotate[i]) + w * rotate[i] <= H + (W + H) * (1 - z[i,b]))
    
    # 6. Non-overlapping constraints with correct auxiliary variables logic
    M = W + H  # Big-M value
    
    for i in range(n):
        for j in range(i+1, n):
            wi, hi = rectangles[i]
            wj, hj = rectangles[j]
            
            for b in range(max_bins):
                # At least one separation must be true if both items are in bin b
                solver.Add(left[i,j,b] + right[i,j,b] + below[i,j,b] + above[i,j,b] >= 
                          z[i,b] + z[j,b] - 1)
                
                # Position constraints based on separation choices (with rotation)
                # i to left of j: x[i] + width_i <= x[j]
                solver.Add(x[i,b] + wi * (1 - rotate[i]) + hi * rotate[i] <= 
                          x[j,b] + M * (1 - left[i,j,b]))
                
                # i to right of j: x[j] + width_j <= x[i]
                solver.Add(x[j,b] + wj * (1 - rotate[j]) + hj * rotate[j] <= 
                          x[i,b] + M * (1 - right[i,j,b]))
                
                # i below j: y[i] + height_i <= y[j]
                solver.Add(y[i,b] + hi * (1 - rotate[i]) + wi * rotate[i] <= 
                          y[j,b] + M * (1 - below[i,j,b]))
                
                # i above j: y[j] + height_j <= y[i]
                solver.Add(y[j,b] + hj * (1 - rotate[j]) + wj * rotate[j] <= 
                          y[i,b] + M * (1 - above[i,j,b]))
    
    # 7. Same-sized rectangles symmetry breaking
    for i in range(n):
        for j in range(i+1, n):
            wi, hi = rectangles[i]
            wj, hj = rectangles[j]
            
            # For identical rectangles (considering rotation)
            if (wi == wj and hi == hj) or (wi == hj and hi == wj):
                # Apply ordering: i must come before j
                for b in range(max_bins):
                    for b2 in range(b):
                        # If i is in bin b and j is in bin b2, then b < b2 is invalid
                        solver.Add(z[i,b] + z[j,b2] <= 1)
                
                # If both in same bin, impose ordering
                for b in range(max_bins):
                    # Either i is to the left of j, or they're at same x and i is below j
                    solver.Add(left[i,j,b] >= z[i,b] + z[j,b] - 1)
    
    # 8. One Pair Constraint (similar to C2 from SPP)
    if n >= 2:
        # Rectangle 1 cannot be to the left of rectangle 0
        solver.Add(left[1,0,0] == 0)
        # Rectangle 1 cannot be below rectangle 0
        solver.Add(below[1,0,0] == 0)
    
    # Set objective: minimize number of bins used
    solver.Minimize(sum(u[b] for b in range(max_bins)))
    
    print(f"Model created in {time.time() - start_model_time:.2f}s")
    
    # Save checkpoint before solving
    save_checkpoint(instance_id, best_bins if best_bins != float('inf') else upper_bound)
    
    # Solve
    print("Solving model...")
    solve_start = time.time()
    status = solver.Solve()
    solve_time = time.time() - solve_start
    
    print(f"Solver finished in {solve_time:.2f}s with status: {status}")
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # Count bins actually used
        bins_used = sum(1 for b in range(max_bins) if u[b].solution_value() > 0.5)
        
        # Update best solution
        if bins_used < best_bins:
            best_bins = bins_used
            
            # Extract item assignments, positions, and rotations
            assignments = [-1] * n
            positions = [(0, 0)] * n
            rotations = [False] * n
            
            for i in range(n):
                rotations[i] = rotate[i].solution_value() > 0.5
                for b in range(max_bins):
                    if z[i,b].solution_value() > 0.5:
                        assignments[i] = b
                        positions[i] = (x[i,b].solution_value(), y[i,b].solution_value())
                        break
            
            best_assignments = assignments.copy()
            best_positions = positions.copy()
            best_rotations = rotations.copy()
            
            # Save checkpoint with solution
            save_checkpoint(instance_id, best_bins)
        
        result = {
            'status': 'COMPLETE' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE',
            'n_bins': bins_used,
            'assignments': assignments,
            'positions': positions,
            'rotations': rotations,
            'solve_time': solve_time,
            'objective_value': solver.Objective().Value()
        }
        
        return result
    else:
        print("No solution found")
        return None

if __name__ == "__main__":
    # Controller mode
    if len(sys.argv) == 1:
        # Create output folder if it doesn't exist
        if not os.path.exists('OR-TOOLS_MIP_R_SB'):
            os.makedirs('OR-TOOLS_MIP_R_SB')
        
        # Read existing Excel file to check completed instances
        excel_file = 'OR-TOOLS_MIP_R_SB.xlsx'
        completed_instances = []
        if os.path.exists(excel_file):
            try:
                existing_df = pd.read_excel(excel_file)
                completed_instances = existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else []
            except:
                existing_df = pd.DataFrame()
                completed_instances = []
        else:
            existing_df = pd.DataFrame()
            completed_instances = []
        
        # Set timeout
        TIMEOUT = 900  # 30 minutes
        
        # Start from instance 1 (skip index 0 which is empty)
        for instance_id in range(1, len(instances)):
            instance_name = instances[instance_id]
            
            # # Skip if already completed
            if instance_name in completed_instances:
                print(f"\nSkipping instance {instance_id}: {instance_name} (already completed)")
                continue
                
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name}")
            print(f"{'=' * 50}")
            
            # Clean up previous result files
            for temp_file in [f'results_OR-TOOLS_MIP_R_SB_{instance_id}.json', f'checkpoint_OR-TOOLS_MIP_R_SB_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with runlim
            command = f"./runlim -r {TIMEOUT} python3 OR-TOOLS_MIP_R_SB.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None
                
                if os.path.exists(f'results_OR-TOOLS_MIP_R_SB_{instance_id}.json'):
                    with open(f'results_OR-TOOLS_MIP_R_SB_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                elif os.path.exists(f'checkpoint_OR-TOOLS_MIP_R_SB_{instance_id}.json'):
                    with open(f'checkpoint_OR-TOOLS_MIP_R_SB_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                    result['Status'] = 'TIMEOUT'
                    result['Instance'] = instance_name
                    print(f"Instance {instance_name} timed out. Using checkpoint data.")
                
                # Process results
                if result:
                    print(f"Instance {instance_name} - Status: {result['Status']}")
                    print(f"Bins used: {result['N_Bins']}, Runtime: {result['Runtime']}")
                    
                    # Update Excel
                    if os.path.exists(excel_file):
                        try:
                            existing_df = pd.read_excel(excel_file)
                            instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                            
                            if instance_exists:
                                instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                                for key, value in result.items():
                                    existing_df.at[instance_idx, key] = value
                            else:
                                result_df = pd.DataFrame([result])
                                existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                        except:
                            existing_df = pd.DataFrame([result])
                    else:
                        existing_df = pd.DataFrame([result])
                    
                    existing_df.to_excel(excel_file, index=False)
                    print(f"Results saved to {excel_file}")
                else:
                    print(f"No results found for instance {instance_name}")
                    
            except Exception as e:
                print(f"Error running instance {instance_name}: {str(e)}")
            
            # Clean up temp files
            for temp_file in [f'results_OR-TOOLS_MIP_R_SB_{instance_id}.json', f'checkpoint_OR-TOOLS_MIP_R_SB_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print(f"\nAll instances completed. Results saved to {excel_file}")
    
    # Single instance mode
    else:
        instance_id = int(sys.argv[1])
        instance_name = instances[instance_id]
        
        start = timeit.default_timer()
        
        try:
            print(f"\nProcessing instance {instance_name}")
            
            # Reset global variables
            best_bins = float('inf')
            best_assignments = []
            best_positions = []
            best_rotations = []
            
            # Read input
            input_data = read_file_instance(instance_name)
            n_items = int(input_data[0])
            bin_size = input_data[1].split()
            W = int(bin_size[0])
            H = int(bin_size[1])
            rectangles = []
            for i in range(2, 2 + n_items):
                line = input_data[i].split()
                demand = int(line[2])
                for _ in range(demand):
                    w = int(line[0])
                    h = int(line[1])
                    rectangles.append((w, h))
            # Calculate bounds
            lower_bound = calculate_lower_bound(rectangles, W, H)
            upper_bound = min(n_items, first_fit_upper_bound(rectangles, W, H, allow_rotation=True))
            
            print(f"Solving 2D Bin Packing with OR-Tools MIP, rotation and symmetry breaking for instance {instance_name}")
            print(f"Bin size: {W}x{H}")
            print(f"Number of items: {n_items}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            
            # Solve with MIP
            solution = solve_bin_packing(W, H, rectangles, lower_bound, upper_bound, time_limit=900)
            
            stop = timeit.default_timer()
            runtime = stop - start
            
            # Process result
            if solution:
                n_bins = solution['n_bins']
                status = 'COMPLETE' if solution['status'] == 'COMPLETE' else 'FEASIBLE'
                
                # Display solution
                display_solution(W, H, rectangles, solution['positions'], solution['assignments'], 
                               solution['rotations'], instance_name)
                
                print(f"Solution found: {n_bins} bins, Status: {status}")
            else:
                n_bins = best_bins if best_bins != float('inf') else upper_bound
                status = 'ERROR'
                print(f"No solution found. Using best bound: {n_bins}")
            
            # Create result
            result = {
                'Instance': instance_name,
                'Runtime': runtime,
                'N_Bins': n_bins,
                'Status': status,
                'Allow_Rotation': 'Yes'
            }
            
            # Save to Excel
            excel_file = 'OR-TOOLS_MIP_R_SB.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except:
                    existing_df = pd.DataFrame([result])
            else:
                existing_df = pd.DataFrame([result])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Results saved to {excel_file}")
            
            # Save JSON result for controller
            with open(f'results_OR-TOOLS_MIP_R_SB_{instance_id}.json', 'w') as f:
                json.dump(result, f)
            
            print(f"Instance {instance_name} completed - Runtime: {runtime:.2f}s, Bins: {n_bins}")

        except Exception as e:
            print(f"Error in instance {instance_name}: {str(e)}")
            traceback.print_exc()
            
            # Create error result
            result = {
                'Instance': instance_name,
                'Runtime': timeit.default_timer() - start,
                'N_Bins': best_bins if best_bins != float('inf') else upper_bound,
                'Status': 'ERROR',
                'Allow_Rotation': 'Yes'
            }
            
            # Save error result to Excel
            excel_file = 'OR-TOOLS_MIP_R_SB.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except:
                    existing_df = pd.DataFrame([result])
            else:
                existing_df = pd.DataFrame([result])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Error results saved to {excel_file}")
            
            with open(f'results_OR-TOOLS_MIP_R_SB_{instance_id}.json', 'w') as f:
                json.dump(result, f)