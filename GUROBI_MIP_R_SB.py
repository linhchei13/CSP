"""
2D Bin Packing Problem - Non-Stacking Version with MIP using Gurobi
With Rotation and Symmetry Breaking constraints
"""

from gurobipy import Model, GRB, quicksum
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
    # Avoid print statements in signal handlers to prevent reentrant call errors
    
    current_bins = best_bins if best_bins != float('inf') else upper_bound
    
    # Save result as JSON for the controller to pick up
    result = {
        'Instance': instances[instance_id],
        'Runtime': timeit.default_timer() - start,
        'N_Bins': current_bins,
        'Status': 'TIMEOUT'
    }
    
    try:
        with open(f'results_GUROBI_MIP_R_{instance_id}.json', 'w') as f:
            json.dump(result, f)
    except:
        pass  # Silently fail if cannot write file
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)  # Termination signal
signal.signal(signal.SIGINT, handle_interrupt)   # Keyboard interrupt (Ctrl+C)

# Create output folder if it doesn't exist
if not os.path.exists('GUROBI_MIP_R_SB'):
    os.makedirs('GUROBI_MIP_R_SB')


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

# Updated instance list with all 510 instances
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


def first_fit_upper_bound_with_rotation(rectangles, W, H):
    """First-fit heuristic to get upper bound with rotation consideration"""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    bins = []
    
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                rect = (x, y, w, h)
                overlap = False
                for (px, py, pw, ph) in bin_rects:
                    if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
        return None
    
    for rect in rectangles:
        w, h = rect[0], rect[1]
        placed = False
        
        # Try both orientations (original and rotated)
        orientations = [(w, h), (h, w)] if w != h else [(w, h)]
        
        for ow, oh in orientations:
            if ow > W or oh > H:
                continue
                
            # Try to place in existing bins
            for bin_rects in bins:
                pos = fits(bin_rects, ow, oh, W, H)
                if pos is not None:
                    bin_rects.append((pos[0], pos[1], ow, oh))
                    placed = True
                    break
            
            if placed:
                break
        
        # If not placed, create a new bin
        if not placed:
            # Try both orientations for new bin
            for ow, oh in orientations:
                if ow <= W and oh <= H:
                    bins.append([(0, 0, ow, oh)])
                    placed = True
                    break
            
            if not placed:
                # Rectangle doesn't fit in any orientation
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
        'Instance': instances[instance_id],
        'Runtime': timeit.default_timer() - start,
        'N_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status
    }

    with open(f'checkpoint_GUROBI_MIP_R_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

def display_solution(W, H, rectangles, positions, assignments, rotations, instance_name):
    """Display solution with one subplot per bin, showing rotation status"""
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
    
    # Use colormap for better visualization
    colors = cm.viridis(np.linspace(0, 1, n_rectangles))
    
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
                width, height = rectangles[item_idx]
                
                # Apply rotation if needed
                if rotations[item_idx] >= 0.5:
                    width, height = height, width
                
                x, y = positions[item_idx]
                
                rect = plt.Rectangle((x, y), width, height, 
                                   edgecolor='black', 
                                   facecolor=colors[item_idx],
                                   alpha=0.7)
                ax.add_patch(rect)
                
                # Calculate text color based on background brightness
                rgb = mcolors.to_rgb(colors[item_idx])
                brightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                text_color = 'white' if brightness < 0.6 else 'black'
                
                # Add item number and rotation status
                rot_info = 'R' if rotations[item_idx] >= 0.5 else 'NR'
                ax.text(x + width/2, y + height/2, f'{item_idx + 1}\n{rot_info}', 
                       ha='center', va='center', fontweight='bold', color=text_color)
            
            # Set grid and ticks
            ax.set_xticks(range(0, W+1, max(1, W//10)))
            ax.set_yticks(range(0, H+1, max(1, H//10)))
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f'GUROBI_MIP_R_SB/{instance_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def solve_bin_packing_with_rotation(W, H, rectangles, lower_bound, upper_bound, time_limit=600):
    """
    Solve 2D Bin Packing using Gurobi MIP with rotation and symmetry breaking
    
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
    
    # Create model
    mdl = Model("2D_BinPacking_Rotation_SB")
    mdl.setParam('OutputFlag', 1)  # Enable output logs
    mdl.setParam('TimeLimit', time_limit)  # Set time limit in seconds
    mdl.setParam('MIPGap', 0.001)  # 0.1% relative gap
    mdl.setParam('MIPGapAbs', 0.01)  # 0.01 absolute gap
    
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
            x[i,b] = mdl.addVar(lb=0, ub=W - rectangles[i][0], vtype=GRB.INTEGER, name=f'x_{i}_{b}')
            y[i,b] = mdl.addVar(lb=0, ub=H - rectangles[i][1], vtype=GRB.INTEGER, name=f'y_{i}_{b}')
    
    # 2. Assignment variables
    z = {}  # z[i,b] = 1 if item i is assigned to bin b
    for i in range(n):
        for b in range(max_bins):
            z[i,b] = mdl.addVar(vtype=GRB.BINARY, name=f'z_{i}_{b}')
    
    # 3. Bin usage variables
    u = {}  # u[b] = 1 if bin b is used
    for b in range(max_bins):
        u[b] = mdl.addVar(vtype=GRB.BINARY, name=f'u_{b}')
    
    # 4. Rotation variables
    rotate = mdl.addVars(n, vtype=GRB.BINARY, name="rotate")
    
    # 3. Auxiliary variables for non-overlap (following debug logic)
    # Create auxiliary variables for all pairs (i,j,b) where i != j
    left = mdl.addVars([(i,j,b) for i in range(n) for j in range(n) if i != j for b in range(max_bins)], 
                       vtype=GRB.BINARY, name="left")   # left[i,j,b] = 1 if item i is to the left of item j in bin b
    right = mdl.addVars([(i,j,b) for i in range(n) for j in range(n) if i != j for b in range(max_bins)], 
                        vtype=GRB.BINARY, name="right")  # right[i,j,b] = 1 if item i is to the right of item j in bin b
    below = mdl.addVars([(i,j,b) for i in range(n) for j in range(n) if i != j for b in range(max_bins)], 
                        vtype=GRB.BINARY, name="below")  # below[i,j,b] = 1 if item i is below item j in bin b
    above = mdl.addVars([(i,j,b) for i in range(n) for j in range(n) if i != j for b in range(max_bins)], 
                        vtype=GRB.BINARY, name="above")  # above[i,j,b] = 1 if item i is above item j in bin b
    
    # Constraints
    
    # 1. Each item must be placed in exactly one bin
    for i in range(n):
        mdl.addConstr(quicksum(z[i,b] for b in range(max_bins)) == 1, f"assign_{i}")
    
    # 2. Bin usage constraints
    for b in range(max_bins):
        for i in range(n):
            mdl.addConstr(z[i,b] <= u[b], f"bin_usage_{i}_{b}")
    
    # 3. Position bounds (considering rotation)
    M = W + H  # Big-M value
    for i in range(n):
        wi, hi = rectangles[i]
        for b in range(max_bins):
            # Width constraint considering rotation
            mdl.addConstr(x[i,b] + wi * (1 - rotate[i]) + hi * rotate[i] <= W + M * (1 - z[i,b]), 
                         f"width_bound_{i}_{b}")
            # Height constraint considering rotation
            mdl.addConstr(y[i,b] + hi * (1 - rotate[i]) + wi * rotate[i] <= H + M * (1 - z[i,b]), 
                         f"height_bound_{i}_{b}")
    
    # 4. Symmetry Breaking: Bins are used in order
    for b in range(1, max_bins):
        mdl.addConstr(u[b] <= u[b-1], f"bin_order_{b}")
    
    # 5. Symmetry Breaking: Largest rectangle placement
    max_area_idx = 0
    max_area = rectangles[0][0] * rectangles[0][1]
    for i in range(1, n):
        area = rectangles[i][0] * rectangles[i][1]
        if area > max_area:
            max_area = area
            max_area_idx = i
    
    if n > 1:
        mdl.addConstr(z[max_area_idx, 0] == 1, "largest_first_bin")
        
        # Domain reduction for largest rectangle
        mdl.addConstr(x[max_area_idx, 0] <= (W - min(rectangles[max_area_idx])) / 2, 
                     "largest_domain_x")
        mdl.addConstr(y[max_area_idx, 0] <= (H - min(rectangles[max_area_idx])) / 2, 
                     "largest_domain_y")
    
    # 6. Non-overlapping constraints with correct auxiliary variables logic
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            wi, hi = rectangles[i]
            wj, hj = rectangles[j]
            
            for b in range(max_bins):
                # At least one separation must be true if both items are in bin b
                mdl.addConstr(left[i,j,b] + right[i,j,b] + below[i,j,b] + above[i,j,b] >= 
                            z[i,b] + z[j,b] - 1, f"separation_{i}_{j}_{b}")
                
                # Position constraints based on separation choices (with rotation)
                # i to left of j: x[i] + width_i <= x[j]
                mdl.addConstr(x[i,b] + wi * (1 - rotate[i]) + hi * rotate[i] <= 
                            x[j,b] + M * (1 - left[i,j,b]), f"left_{i}_{j}_{b}")
                
                # i to right of j: x[j] + width_j <= x[i]
                mdl.addConstr(x[j,b] + wj * (1 - rotate[j]) + hj * rotate[j] <= 
                            x[i,b] + M * (1 - right[i,j,b]), f"right_{i}_{j}_{b}")
                
                # i below j: y[i] + height_i <= y[j]
                mdl.addConstr(y[i,b] + hi * (1 - rotate[i]) + wi * rotate[i] <= 
                            y[j,b] + M * (1 - below[i,j,b]), f"below_{i}_{j}_{b}")
                
                # i above j: y[j] + height_j <= y[i]
                mdl.addConstr(y[j,b] + hj * (1 - rotate[j]) + wj * rotate[j] <= 
                            y[i,b] + M * (1 - above[i,j,b]), f"above_{i}_{j}_{b}")
    
    # 7. Same-sized rectangles symmetry breaking (following OR-TOOLS logic)
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
                        mdl.addConstr(z[i,b] + z[j,b2] <= 1, f"same_rect_bin_{i}_{j}_{b}_{b2}")
                
                # If both in same bin, impose ordering
                for b in range(max_bins):
                    # Either i is to the left of j, or they're at same x and i is below j
                    mdl.addConstr(left[i,j,b] >= z[i,b] + z[j,b] - 1, f"same_rect_order_{i}_{j}_{b}")
    
    # 8. One Pair Constraint (similar to C2 from SPP, following OR-TOOLS logic)
    if n >= 2:
        # Rectangle 1 cannot be to the left of rectangle 0
        mdl.addConstr(left[1,0,0] == 0, f"pair_left_0")
        # Rectangle 1 cannot be below rectangle 0
        mdl.addConstr(below[1,0,0] == 0, f"pair_below_0")
    
    # Set objective: minimize number of bins used
    mdl.setObjective(quicksum(u[b] for b in range(max_bins)), GRB.MINIMIZE)
    
    print(f"Model created in {time.time() - start_model_time:.2f}s")
    
    # Save checkpoint before solving
    save_checkpoint(instance_id, best_bins if best_bins != float('inf') else upper_bound)
    
    # Optimize
    print("Solving model...")
    solve_start = time.time()
    mdl.optimize()
    solve_time = time.time() - solve_start
    
    print(f"Solver finished in {solve_time:.2f}s with status: {mdl.status}")
    
    if mdl.status == GRB.OPTIMAL or (mdl.status == GRB.TIME_LIMIT and mdl.solCount > 0):
        # Count bins actually used
        bins_used = sum(1 for b in range(max_bins) if u[b].x > 0.5)
        
        # Update best solution
        if bins_used < best_bins:
            best_bins = bins_used
            
            # Extract item assignments, positions, and rotations
            assignments = [-1] * n
            positions = [(0, 0)] * n
            rotations = [0] * n
            
            for i in range(n):
                rotations[i] = rotate[i].x
                for b in range(max_bins):
                    if z[i,b].x > 0.5:
                        assignments[i] = b
                        positions[i] = (x[i,b].x, y[i,b].x)
                        break
            
            best_assignments = assignments.copy()
            best_positions = positions.copy()
            best_rotations = rotations.copy()
            
            # Save checkpoint with solution
            save_checkpoint(instance_id, best_bins)
        
        result = {
            'status': 'OPTIMAL' if mdl.status == GRB.OPTIMAL else 'FEASIBLE',
            'n_bins': bins_used,
            'assignments': assignments,
            'positions': positions,
            'rotations': rotations,
            'solve_time': solve_time,
            'objective_value': mdl.objVal,
            'gap': mdl.MIPGap * 100 if hasattr(mdl, 'MIPGap') else None
        }
        
        return result
    else:
        print("No solution found")
        return None

if __name__ == "__main__":
    # Controller mode
    if len(sys.argv) == 1:
        # Create output folder if it doesn't exist
        if not os.path.exists('GUROBI_MIP_R_SB'):
            os.makedirs('GUROBI_MIP_R_SB')
        
        # Read existing Excel file to check completed instances
        excel_file = 'GUROBI_MIP_R_SB.xlsx'
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
            
            # Skip if already completed
            if instance_name in completed_instances:
                print(f"\nSkipping instance {instance_id}: {instance_name} (already completed)")
                continue
                
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name}")
            print(f"{'=' * 50}")
            
            # Clean up previous result files
            for temp_file in [f'results_GUROBI_MIP_R_{instance_id}.json', f'checkpoint_GUROBI_MIP_R_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with runlim
            command = f"./runlim -r {TIMEOUT} python3 GUROBI_MIP_R_SB.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None
                
                if os.path.exists(f'results_GUROBI_MIP_R_{instance_id}.json'):
                    with open(f'results_GUROBI_MIP_R_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                elif os.path.exists(f'checkpoint_GUROBI_MIP_R_{instance_id}.json'):
                    with open(f'checkpoint_GUROBI_MIP_R_{instance_id}.json', 'r') as f:
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
            for temp_file in [f'results_GUROBI_MIP_R_{instance_id}.json', f'checkpoint_GUROBI_MIP_R_{instance_id}.json']:
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
                w = int(line[0])
                h = int(line[1])
                demand = int(line[2])  # Unused in current model
                for _ in range(demand):
                    rectangles.append((w, h))
            # Calculate bounds
            lower_bound = calculate_lower_bound(rectangles, W, H)
            upper_bound = min(n_items, first_fit_upper_bound_with_rotation(rectangles, W, H))
            
            print(f"Solving 2D Bin Packing with Gurobi MIP (with rotation and symmetry breaking) for instance {instance_name}")
            print(f"Bin size: {W}x{H}")
            print(f"Number of items: {n_items}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            
            # Solve with MIP
            solution = solve_bin_packing_with_rotation(W, H, rectangles, lower_bound, upper_bound, time_limit=900)
            
            stop = timeit.default_timer()
            runtime = stop - start
            
            # Process result
            if solution:
                n_bins = solution['n_bins']
                status = 'OPTIMAL' if solution['status'] == 'OPTIMAL' else 'FEASIBLE'
                
                # Display solution
                display_solution(W, H, rectangles, solution['positions'], solution['assignments'], 
                               solution['rotations'], instance_name)
                
                print(f"Solution found: {n_bins} bins, Status: {status}")
                if solution['gap'] is not None:
                    print(f"Optimality gap: {solution['gap']:.2f}%")
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
            }
            
            # Save to Excel
            excel_file = 'GUROBI_MIP_R_SB.xlsx'
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
            with open(f'results_GUROBI_MIP_R_{instance_id}.json', 'w') as f:
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
            excel_file = 'GUROBI_MIP_R_SB.xlsx'
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
            
            with open(f'results_GUROBI_MIP_R_{instance_id}.json', 'w') as f:
                json.dump(result, f)