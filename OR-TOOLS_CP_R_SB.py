"""
2D Bin Packing Problem - Rotation Version with CP using OR-Tools
Symmetry Breaking: C1 (Bins are used in order)
"""

from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import timeit
import sys
import signal
import os
import json
import subprocess
import traceback
import pandas as pd

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
        'Status': 'TIMEOUT'
    }
    
    with open(f'results_OR-TOOLS_CP_R_SB_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

# Create output folder if it doesn't exist
if not os.path.exists('OR-TOOLS_CP_R_SB'):
    os.makedirs('OR-TOOLS_CP_R_SB')

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
    """First-fit heuristic to get upper bound with rotation allowed"""
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
        
        # Try both orientations
        orientations = [(w, h), (h, w)] if w != h else [(w, h)]
        
        for ow, oh in orientations:
            if ow <= W and oh <= H:
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
            best_orient = None
            for ow, oh in orientations:
                if ow <= W and oh <= H:
                    best_orient = (ow, oh)
                    break
            
            if best_orient:
                bins.append([(0, 0, best_orient[0], best_orient[1])])
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
        'Runtime': timeit.default_timer() - start,
        'N_Bins': bins if bins != float('inf') else upper_bound,
        'Status': status
    }
    
    with open(f'checkpoint_OR-TOOLS_CP_R_SB_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """Callback to track best solutions found during solving"""
    
    def __init__(self, u_vars, z_vars, x_vars, y_vars, rotate_vars, n_items, max_bins):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.u_vars = u_vars
        self.z_vars = z_vars
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.rotate_vars = rotate_vars
        self.n_items = n_items
        self.max_bins = max_bins
        self.solution_count = 0
        
    def on_solution_callback(self):
        global best_bins, best_assignments, best_positions, best_rotations
        
        self.solution_count += 1
        
        # Count bins used
        bins_used = sum(1 for b in range(self.max_bins) if self.Value(self.u_vars[b]) == 1)
        
        # Update best solution if better
        if bins_used < best_bins:
            best_bins = bins_used
            
            # Extract assignments, positions, and rotations
            assignments = [-1] * self.n_items
            positions = [(0, 0)] * self.n_items
            rotations = [0] * self.n_items
            
            for i in range(self.n_items):
                for b in range(self.max_bins):
                    if self.Value(self.z_vars[i,b]) == 1:
                        assignments[i] = b
                        positions[i] = (self.Value(self.x_vars[i,b]), self.Value(self.y_vars[i,b]))
                        rotations[i] = self.Value(self.rotate_vars[i])
                        break
            
            best_assignments = assignments.copy()
            best_positions = positions.copy()
            best_rotations = rotations.copy()
            
            # Save checkpoint with better solution
            save_checkpoint(instance_id, best_bins)
            
            print(f"  New best solution found: {bins_used} bins (solution #{self.solution_count})")

def solve_bin_packing_with_rotation(W, H, rectangles, time_limit=600):
    """
    Solves the 2D Bin Packing Problem with rotation using OR-Tools CP-SAT with objective minimization.
    
    Args:
        W: Width of each bin
        H: Height of each bin
        rectangles: List of (width, height) tuples for each rectangle
        time_limit: Time limit for the solver in seconds
        
    Returns:
        dict with optimal number of bins, assignments, positions and rotations
    """
    global best_bins, best_assignments, best_positions, best_rotations, upper_bound
    
    # Create the model
    model = cp_model.CpModel()
    
    n = len(rectangles)
    
    # Calculate upper bound
    ub = min(n, first_fit_upper_bound_with_rotation(rectangles, W, H))
    upper_bound = ub
    max_bins = ub  # Use upper bound as maximum number of bins
    
    print(f"Creating CP model with {n} items and up to {max_bins} bins...")
    start_model_time = time.time()
    
    # Variables: coordinates of the bottom-left corner of each rectangle
    x = {}
    y = {}
    z = {}  # z[i,b] = 1 if item i is assigned to bin b
    rotate = {}  # rotate[i] = 1 if item i is rotated
    
    for i in range(n):
        rotate[i] = model.NewBoolVar(f'rotate_{i}')
        for b in range(max_bins):
            # Max possible width/height considering rotation
            max_w = max(rectangles[i][0], rectangles[i][1])
            max_h = max(rectangles[i][0], rectangles[i][1])
            
            x[i,b] = model.NewIntVar(0, W - min(rectangles[i][0], rectangles[i][1]), f'x_{i}_{b}')
            y[i,b] = model.NewIntVar(0, H - min(rectangles[i][0], rectangles[i][1]), f'y_{i}_{b}')
            z[i,b] = model.NewBoolVar(f'z_{i}_{b}')
    
    # Variables for actual width and height after rotation
    actual_width = {}
    actual_height = {}
    
    for i in range(n):
        w_orig, h_orig = rectangles[i]
        
        actual_width[i] = model.NewIntVar(min(w_orig, h_orig), max(w_orig, h_orig), f'width_{i}')
        actual_height[i] = model.NewIntVar(min(w_orig, h_orig), max(w_orig, h_orig), f'height_{i}')
        
        # Set actual dimensions based on rotation
        model.Add(actual_width[i] == w_orig).OnlyEnforceIf(rotate[i].Not())
        model.Add(actual_width[i] == h_orig).OnlyEnforceIf(rotate[i])
        model.Add(actual_height[i] == h_orig).OnlyEnforceIf(rotate[i].Not())
        model.Add(actual_height[i] == w_orig).OnlyEnforceIf(rotate[i])
    
    # Bin usage variables
    u = {}  # u[b] = 1 if bin b is used
    for b in range(max_bins):
        u[b] = model.NewBoolVar(f'u_{b}')
    
    # Each item must be placed in exactly one bin
    for i in range(n):
        model.Add(sum(z[i,b] for b in range(max_bins)) == 1)
    
    # Bin usage constraints
    for b in range(max_bins):
        for i in range(n):
            model.Add(z[i,b] <= u[b])
    
    # Boundary constraints considering rotation
    for i in range(n):
        for b in range(max_bins):
            model.Add(x[i,b] + actual_width[i] <= W).OnlyEnforceIf(z[i,b])
            model.Add(y[i,b] + actual_height[i] <= H).OnlyEnforceIf(z[i,b])
    
    # C1 Symmetry Breaking: Bins are used in order
    for b in range(1, max_bins):
        model.Add(u[b] <= u[b-1])
    
    # Find largest rectangle by area for additional symmetry breaking
    max_area_idx = 0
    max_area = rectangles[0][0] * rectangles[0][1]
    for i in range(1, n):
        area = rectangles[i][0] * rectangles[i][1]
        if area > max_area:
            max_area = area
            max_area_idx = i
    
    # C1 additional symmetry breaking: Place largest rectangle in first bin
    if n > 1:
        model.Add(z[max_area_idx, 0] == 1)
        
        # Position the largest rectangle in the bottom-left quadrant
        model.Add(x[max_area_idx, 0] <= (W - min(rectangles[max_area_idx])) // 2)
        model.Add(y[max_area_idx, 0] <= (H - min(rectangles[max_area_idx])) // 2)
    
    # Non-overlapping constraints within each bin
    for b in range(max_bins):
        for i in range(n):
            for j in range(i + 1, n):
                # Define Boolean variables for the four possible arrangements
                b_left_i_j = model.NewBoolVar(f'b_left_{i}_{j}_{b}')
                b_left_j_i = model.NewBoolVar(f'b_left_{j}_{i}_{b}')
                b_below_i_j = model.NewBoolVar(f'b_below_{i}_{j}_{b}')
                b_below_j_i = model.NewBoolVar(f'b_below_{j}_{i}_{b}')
                
                # Position constraints using actual dimensions
                model.Add(x[i,b] + actual_width[i] <= x[j,b]).OnlyEnforceIf(b_left_i_j)
                model.Add(x[j,b] + actual_width[j] <= x[i,b]).OnlyEnforceIf(b_left_j_i)
                model.Add(y[i,b] + actual_height[i] <= y[j,b]).OnlyEnforceIf(b_below_i_j)
                model.Add(y[j,b] + actual_height[j] <= y[i,b]).OnlyEnforceIf(b_below_j_i)
                
                # If both items are in this bin, they must not overlap
                both_in_bin = model.NewBoolVar(f'both_in_bin_{i}_{j}_{b}')
                model.Add(z[i,b] + z[j,b] == 2).OnlyEnforceIf(both_in_bin)
                model.Add(z[i,b] + z[j,b] <= 1).OnlyEnforceIf(both_in_bin.Not())
                
                # When both items are in the same bin, at least one arrangement must be true
                model.Add(b_left_i_j + b_left_j_i + b_below_i_j + b_below_j_i >= 1).OnlyEnforceIf(both_in_bin)
                
                # C1 Symmetry Breaking - Large rectangles with rotation
                w_i, h_i = rectangles[i]
                w_j, h_j = rectangles[j]
                
                # Minimum possible width after rotation for both rectangles
                min_width_i = min(w_i, h_i)
                min_width_j = min(w_j, h_j)
                
                if min_width_i + min_width_j > W:
                    # If two rectangles can't fit side by side even with rotation, disable horizontal
                    model.Add(b_left_i_j == 0).OnlyEnforceIf(both_in_bin)
                    model.Add(b_left_j_i == 0).OnlyEnforceIf(both_in_bin)
                
                # Minimum possible height after rotation for both rectangles
                min_height_i = min(w_i, h_i)
                min_height_j = min(w_j, h_j)
                
                if min_height_i + min_height_j > H:
                    # If two rectangles can't fit vertically even with rotation, disable vertical
                    model.Add(b_below_i_j == 0).OnlyEnforceIf(both_in_bin)
                    model.Add(b_below_j_i == 0).OnlyEnforceIf(both_in_bin)
    
    # Same-sized rectangles C1 constraint (considering rotation)
    for i in range(n):
        for j in range(i+1, n):
            # Check if rectangles are equivalent (same dimensions possibly rotated)
            w_i, h_i = rectangles[i]
            w_j, h_j = rectangles[j]
            
            equivalent = (w_i == w_j and h_i == h_j) or (w_i == h_j and h_i == w_j)
            
            if equivalent:
                # For equivalent rectangles, apply ordering
                for b in range(max_bins):
                    for b2 in range(b):
                        # If i is in bin b and j is in bin b2, then b < b2 is invalid
                        model.Add(z[i,b] + z[j,b2] <= 1)
                
                # If both in same bin, impose ordering (lexicographic)
                for b in range(max_bins):
                    both_in_bin = model.NewBoolVar(f'equiv_both_in_bin_{i}_{j}_{b}')
                    model.Add(z[i,b] + z[j,b] == 2).OnlyEnforceIf(both_in_bin)
                    model.Add(z[i,b] + z[j,b] <= 1).OnlyEnforceIf(both_in_bin.Not())
                    
                    # i must be to the left of j, or at same x and i below j
                    left_ordering = model.NewBoolVar(f'left_ordering_{i}_{j}_{b}')
                    model.Add(x[i,b] < x[j,b]).OnlyEnforceIf([both_in_bin, left_ordering])
                    
                    same_x = model.NewBoolVar(f'same_x_{i}_{j}_{b}')
                    model.Add(x[i,b] == x[j,b]).OnlyEnforceIf([both_in_bin, same_x, left_ordering.Not()])
                    model.Add(y[i,b] <= y[j,b]).OnlyEnforceIf([both_in_bin, same_x, left_ordering.Not()])
    
    # Set objective: minimize number of bins used
    model.Minimize(sum(u[b] for b in range(max_bins)))
    
    print(f"Model created in {time.time() - start_model_time:.2f}s")
    
    # Save checkpoint before solving
    save_checkpoint(instance_id, best_bins if best_bins != float('inf') else upper_bound)
    
    # Create callback to track solutions
    solution_callback = SolutionCallback(u, z, x, y, rotate, n, max_bins)
    
    # Solve with CP-SAT
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = False  # We want the optimal solution
    
    print("Solving with CP-SAT...")
    solve_start = time.time()
    status = solver.SolveWithSolutionCallback(model, solution_callback)
    solve_time = time.time() - solve_start
    
    print(f"Solver finished in {solve_time:.2f}s with status: {solver.StatusName(status)}")
    print(f"Solutions found: {solution_callback.solution_count}")
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Get final solution
        bins_used = sum(1 for b in range(max_bins) if solver.Value(u[b]) == 1)
        
        # Update best solution if this is better
        if bins_used < best_bins:
            best_bins = bins_used
            
            # Extract final assignments, positions, and rotations
            assignments = [-1] * n
            positions = [(0, 0)] * n
            rotations = [0] * n
            
            for i in range(n):
                for b in range(max_bins):
                    if solver.Value(z[i,b]) == 1:
                        assignments[i] = b
                        positions[i] = (solver.Value(x[i,b]), solver.Value(y[i,b]))
                        rotations[i] = solver.Value(rotate[i])
                        break
            
            best_assignments = assignments.copy()
            best_positions = positions.copy()
            best_rotations = rotations.copy()
            
            # Save final checkpoint
            save_checkpoint(instance_id, best_bins)
        
        return {
            'n_bins': bins_used,
            'assignments': best_assignments if best_assignments else assignments,
            'positions': best_positions if best_positions else positions,
            'rotations': best_rotations if best_rotations else rotations,
            'optimal': status == cp_model.OPTIMAL,
            'solve_time': solve_time,
            'objective_value': solver.ObjectiveValue() if status == cp_model.OPTIMAL else bins_used,
            'solutions_found': solution_callback.solution_count
        }
    else:
        # No solution found, return the upper bound
        return {
            'n_bins': best_bins if best_bins != float('inf') else upper_bound,
            'assignments': best_assignments,
            'positions': best_positions,
            'rotations': best_rotations,
            'optimal': False,
            'solve_time': solve_time,
            'objective_value': None,
            'solutions_found': solution_callback.solution_count
        }

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
                orig_width, orig_height = rectangles[item_idx]
                x, y = positions[item_idx]
                is_rotated = rotations[item_idx]
                
                # Apply rotation
                if is_rotated:
                    width, height = orig_height, orig_width
                else:
                    width, height = orig_width, orig_height
                
                rect = plt.Rectangle((x, y), width, height, 
                                   edgecolor='black', 
                                   facecolor=plt.cm.Set3(item_idx % 12),
                                   alpha=0.7)
                ax.add_patch(rect)
                
                # Add item number and rotation info
                rot_info = 'R' if is_rotated else 'NR'
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
    plt.savefig(f'OR-TOOLS_CP_R_SB/{instance_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Controller mode
    if len(sys.argv) == 1:
        # Create output folder if it doesn't exist
        if not os.path.exists('OR-TOOLS_CP_R_SB'):
            os.makedirs('OR-TOOLS_CP_R_SB')
        
        # Read existing Excel file to check completed instances
        excel_file = 'OR-TOOLS_CP_R_SB.xlsx'
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
        TIMEOUT = 900  # 10 minutes

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
            for temp_file in [f'results_OR-TOOLS_CP_R_SB_{instance_id}.json', f'checkpoint_OR-TOOLS_CP_R_SB_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with runlim
            command = f"./runlim -r {TIMEOUT} python3 OR-TOOLS_CP_R_SB.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None
                
                if os.path.exists(f'results_OR-TOOLS_CP_R_SB_{instance_id}.json'):
                    with open(f'results_OR-TOOLS_CP_R_SB_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                elif os.path.exists(f'checkpoint_OR-TOOLS_CP_R_SB_{instance_id}.json'):
                    with open(f'checkpoint_OR-TOOLS_CP_R_SB_{instance_id}.json', 'r') as f:
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
            for temp_file in [f'results_OR-TOOLS_CP_R_SB_{instance_id}.json', f'checkpoint_OR-TOOLS_CP_R_SB_{instance_id}.json']:
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
            upper_bound = min(n_items, first_fit_upper_bound_with_rotation(rectangles, W, H))
            
            print(f"Solving 2D Bin Packing with OR-Tools CP and C1 symmetry breaking (with rotation) for instance {instance_name}")
            print(f"Bin size: {W}x{H}")
            print(f"Number of items: {n_items}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            
            # Solve with CP
            result = solve_bin_packing_with_rotation(W, H, rectangles, time_limit=900)
            
            stop = timeit.default_timer()
            runtime = stop - start
            
            # Process result
            if len(result['positions']) > 0:
                # Display solution
                display_solution(W, H, rectangles, result['positions'], result['assignments'], result['rotations'], instance_name)
                
                print(f"Solution found: {result['n_bins']} bins")
                print(f"Solutions explored: {result['solutions_found']}")
                status = 'OPTIMAL' if result['optimal'] else 'FEASIBLE'
                
                if result['objective_value'] is not None:
                    print(f"Objective value: {result['objective_value']}")
            else:
                print("No feasible solution found.")
                status = 'ERROR'
            
            # Create result
            result_data = {
                'Instance': instance_name,
                'Runtime': runtime,
                'N_Bins': result['n_bins'],
                'Status': status
            }
            
            # Save to Excel
            excel_file = 'OR-TOOLS_CP_R_SB.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result_data.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result_data])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except:
                    existing_df = pd.DataFrame([result_data])
            else:
                existing_df = pd.DataFrame([result_data])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Results saved to {excel_file}")
            
            # Save JSON result for controller
            with open(f'results_OR-TOOLS_CP_R_SB_{instance_id}.json', 'w') as f:
                json.dump(result_data, f)
            
            print(f"Instance {instance_name} completed - Runtime: {runtime:.2f}s, Bins: {result['n_bins']}")

        except Exception as e:
            print(f"Error in instance {instance_name}: {str(e)}")
            traceback.print_exc()
            
            # Create error result
            result_data = {
                'Instance': instance_name,
                'Runtime': timeit.default_timer() - start,
                'N_Bins': best_bins if best_bins != float('inf') else upper_bound,
                'Status': 'ERROR'
            }
            
            # Save error result to Excel
            excel_file = 'OR-TOOLS_CP_R_SB.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result_data.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result_data])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except:
                    existing_df = pd.DataFrame([result_data])
            else:
                existing_df = pd.DataFrame([result_data])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Error results saved to {excel_file}")
            
            with open(f'results_OR-TOOLS_CP_R_SB_{instance_id}.json', 'w') as f:
                json.dump(result_data, f)