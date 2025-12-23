"""
2D Bin Packing Problem - Non-Stacking Version with MIP using Gurobi
Symmetry Breaking: C1 (Enhanced implementation)
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
        'Status': 'TIMEOUT'
    }
    
    with open(f'results_GUROBI_MIP_C1_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

# Create output folder if it doesn't exist
if not os.path.exists('GUROBI_MIP_C1'):
    os.makedirs('GUROBI_MIP_C1')


def first_fit_upper_bound(rectangles, W, H):
    """First-fit heuristic to get upper bound"""
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
        
        # Try to place in existing bins
        for bin_rects in bins:
            pos = fits(bin_rects, w, h, W, H)
            if pos is not None:
                bin_rects.append((pos[0], pos[1], w, h))
                placed = True
                break
        
        # If not placed, create a new bin
        if not placed:
            if w <= W and h <= H:
                bins.append([(0, 0, w, h)])
            else:
                # Rectangle doesn't fit in any bin
                return float('inf')
    
    return len(bins)

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
    
    with open(f'checkpoint_GUROBI_MIP_C1_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

def display_solution(W, H, rectangles, positions, assignments, instance_name):
    """Display solution with one subplot per bin"""
    n_bins = len(set(assignments))
    n_rectangles = len(rectangles)
    
    # Determine layout of subplots
    ncols = min(n_bins, 3)
    nrows = math.ceil(n_bins / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    fig.suptitle(f'Solution for {instance_name} - {n_bins} bins', fontsize=16)
    
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
                width, height = rectangles[item_idx]
                x, y = positions[item_idx]
                
                rect = plt.Rectangle((x, y), width, height, 
                                   edgecolor='black', 
                                   facecolor=plt.cm.Set3(item_idx % 12),
                                   alpha=0.7)
                ax.add_patch(rect)
                
                # Add item number
                ax.text(x + width/2, y + height/2, str(item_idx + 1), 
                       ha='center', va='center', fontweight='bold')
            
            # Set grid and ticks
            ax.set_xticks(range(0, W+1, max(1, W//10)))
            ax.set_yticks(range(0, H+1, max(1, H//10)))
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f'GUROBI_MIP_C1/{instance_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def solve_bin_packing(W, H, rectangles, lower_bound, upper_bound, time_limit=900):
    """
    Solve 2D Bin Packing using Gurobi MIP with C1 symmetry breaking
    
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
    global best_bins, best_assignments, best_positions
    
    # Create the model
    mdl = Model("2D_BinPacking_C1")
    mdl.setParam('OutputFlag', 1)  # Enable Gurobi output logs
    mdl.setParam('TimeLimit', time_limit)  # Set time limit
    
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
    
    # 4. Auxiliary variables for non-overlapping constraints
    left = {}   # left[i,j,b] = 1 if item i is to the left of item j in bin b
    right = {}  # right[i,j,b] = 1 if item i is to the right of item j in bin b
    below = {}  # below[i,j,b] = 1 if item i is below item j in bin b
    above = {}  # above[i,j,b] = 1 if item i is above item j in bin b
    
    for i in range(n):
        for j in range(i+1, n):
            for b in range(max_bins):
                left[i,j,b] = mdl.addVar(vtype=GRB.BINARY, name=f'left_{i}_{j}_{b}')
                right[i,j,b] = mdl.addVar(vtype=GRB.BINARY, name=f'right_{i}_{j}_{b}')
                below[i,j,b] = mdl.addVar(vtype=GRB.BINARY, name=f'below_{i}_{j}_{b}')
                above[i,j,b] = mdl.addVar(vtype=GRB.BINARY, name=f'above_{i}_{j}_{b}')
    
    # Constraints
    
    # 1. Each item must be placed in exactly one bin
    for i in range(n):
        mdl.addConstr(quicksum(z[i,b] for b in range(max_bins)) == 1, f"assign_{i}")
    
    # 2. Bin usage constraints - if an item is in bin b, bin b must be used
    for b in range(max_bins):
        for i in range(n):
            mdl.addConstr(z[i,b] <= u[b], f"usage_{b}_{i}")
    
    # 3. C1 Symmetry Breaking: Bins are used in order
    for b in range(1, max_bins):
        mdl.addConstr(u[b] <= u[b-1], f"order_{b}")
    
    # Find largest rectangle by area for additional symmetry breaking
    max_area_idx = 0
    max_area = rectangles[0][0] * rectangles[0][1]
    
    for i in range(1, n):
        area = rectangles[i][0] * rectangles[i][1]
        if area > max_area:
            max_area = area
            max_area_idx = i
    
    # 4. C1 additional symmetry breaking: Place largest rectangle in first bin
    if n > 1:
        mdl.addConstr(z[max_area_idx, 0] == 1, "largest_first")
        
        # Position the largest rectangle in the bottom-left quadrant
        mdl.addConstr(x[max_area_idx, 0] <= (W - rectangles[max_area_idx][0]) // 2, "largest_x")
        mdl.addConstr(y[max_area_idx, 0] <= (H - rectangles[max_area_idx][1]) // 2, "largest_y")
    
    # 5. Non-overlapping constraints - Fixed version
    for i in range(n):
        for j in range(i+1, n):
            for b in range(max_bins):
                # If both items i and j are in bin b, they must not overlap
                M = max(W, H)  # Big-M value
                
                # At least one separation must be true if both items are in bin b
                mdl.addConstr(left[i,j,b] + right[i,j,b] + below[i,j,b] + above[i,j,b] >= 
                              z[i,b] + z[j,b] - 1, f"overlap_{i}_{j}_{b}")
                
                # Position constraints based on separation choices
                # i to left of j: x[i] + w[i] <= x[j]
                mdl.addConstr(x[i,b] + rectangles[i][0] <= x[j,b] + M * (1 - left[i,j,b]), f"left_{i}_{j}_{b}")
                
                # i to right of j: x[j] + w[j] <= x[i]  
                mdl.addConstr(x[j,b] + rectangles[j][0] <= x[i,b] + M * (1 - right[i,j,b]), f"right_{i}_{j}_{b}")
                
                # i below j: y[i] + h[i] <= y[j]
                mdl.addConstr(y[i,b] + rectangles[i][1] <= y[j,b] + M * (1 - below[i,j,b]), f"below_{i}_{j}_{b}")
                
                # i above j: y[j] + h[j] <= y[i]
                mdl.addConstr(y[j,b] + rectangles[j][1] <= y[i,b] + M * (1 - above[i,j,b]), f"above_{i}_{j}_{b}")
    
    # 6. Same-sized rectangles C1 constraint
    for i in range(n):
        for j in range(i+1, n):
            if rectangles[i][0] == rectangles[j][0] and rectangles[i][1] == rectangles[j][1]:
                # For identical rectangles, apply ordering: i must come before j
                for b in range(max_bins):
                    for b2 in range(b):
                        # If i is in bin b and j is in bin b2, then b < b2 is invalid
                        mdl.addConstr(z[i,b] + z[j,b2] <= 1, f"identical_{i}_{j}_{b}_{b2}")
                
                # If both in same bin, impose ordering (lexicographic)
                for b in range(max_bins):
                    # Either i is to the left of j, or they're at same x and i is below j
                    mdl.addConstr(left[i,j,b] + below[i,j,b] >= z[i,b] + z[j,b] - 1, f"lex_{i}_{j}_{b}")
    
    # Set objective: minimize number of bins used
    mdl.setObjective(quicksum(u[b] for b in range(max_bins)), GRB.MINIMIZE)
    
    print(f"Model created in {time.time() - start_model_time:.2f}s")
    
    # Save checkpoint before solving
    save_checkpoint(instance_id, best_bins if best_bins != float('inf') else upper_bound)
    
    # Solve
    print("Solving model...")
    solve_start = time.time()
    mdl.optimize()
    solve_time = time.time() - solve_start
    
    print(f"Solver finished in {solve_time:.2f}s with status: {mdl.status}")
    
    if mdl.status == GRB.OPTIMAL or mdl.status == GRB.TIME_LIMIT:
        if mdl.solCount > 0:
            # Count bins actually used
            bins_used = sum(1 for b in range(max_bins) if u[b].X > 0.5)
            
            # Update best solution
            if bins_used < best_bins:
                best_bins = bins_used
                
                # Extract item assignments
                assignments = [-1] * n
                positions = [(0, 0)] * n
                
                for i in range(n):
                    for b in range(max_bins):
                        if z[i,b].X > 0.5:
                            assignments[i] = b
                            positions[i] = (int(x[i,b].X), int(y[i,b].X))
                            break
                
                best_assignments = assignments.copy()
                best_positions = positions.copy()
                
                # Save checkpoint with solution
                save_checkpoint(instance_id, best_bins)
            
            result = {
                'status': 'OPTIMAL' if mdl.status == GRB.OPTIMAL else 'TIMEOUT',
                'n_bins': bins_used,
                'assignments': assignments,
                'positions': positions,
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
        if not os.path.exists('GUROBI_MIP_C1'):
            os.makedirs('GUROBI_MIP_C1')
        
        # Read existing Excel file to check completed instances
        excel_file = 'GUROBI_MIP_C1.xlsx'
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
            for temp_file in [f'results_GUROBI_MIP_C1_{instance_id}.json', f'checkpoint_GUROBI_MIP_C1_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with runlim
            command = f"./runlim -r {TIMEOUT} python3 GUROBI_MIP_C1.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None

                if os.path.exists(f'results_GUROBI_MIP_C1_{instance_id}.json'):
                    with open(f'results_GUROBI_MIP_C1_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                elif os.path.exists(f'checkpoint_GUROBI_MIP_C1_{instance_id}.json'):
                    with open(f'checkpoint_GUROBI_MIP_C1_{instance_id}.json', 'r') as f:
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
            for temp_file in [f'results_GUROBI_MIP_C1_{instance_id}.json', f'checkpoint_GUROBI_MIP_C1_{instance_id}.json']:
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
            upper_bound = min(n_items, first_fit_upper_bound(rectangles, W, H))
            
            print(f"Solving 2D Bin Packing with Gurobi MIP and C1 symmetry breaking for instance {instance_name}")
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
                status = 'OPTIMAL' if solution['status'] == 'OPTIMAL' else 'TIMELIMIT'
                
                # Display solution
                display_solution(W, H, rectangles, solution['positions'], solution['assignments'], instance_name)
                
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
                'Status': status
            }
            
            # Save to Excel
            excel_file = 'GUROBI_MIP_C1.xlsx'
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
            with open(f'results_GUROBI_MIP_C1_{instance_id}.json', 'w') as f:
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
                'Status': 'ERROR'
            }
            
            # Save error result to Excel
            excel_file = 'GUROBI_MIP_C1.xlsx'
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
            
            with open(f'results_GUROBI_MIP_C1_{instance_id}.json', 'w') as f:
                json.dump(result, f)