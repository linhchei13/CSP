import math
import fileinput
import matplotlib.pyplot as plt
import timeit
import sys
import signal
import pandas as pd
import os
import tempfile
import subprocess
import traceback
import json
import time

from pysat.formula import CNF
from pysat.solvers import Glucose42

# Global variables to track best solution found so far
best_num_bins = float('inf')
best_positions = []
best_rotations = []
variables_length = 0
clauses_length = 0
upper_bound = 0  # Biến toàn cục để lưu trữ upper_bound
wcnf_file = None  # Biến toàn cục để lưu trữ tên file wcnf

# Signal handler for graceful interruption (e.g., by runlim)
def handle_interrupt(signum, frame):
    print(f"\nReceived interrupt signal {signum}. Saving current best solution.")
    
    # Lấy số bins tốt nhất (hoặc là giá trị tìm được, hoặc là upper_bound)
    current_bins = best_num_bins if best_num_bins != float('inf') else upper_bound
    print(f"Best number of bins found before interrupt: {current_bins}")
    
    # Save result as JSON for the controller to pick up
    result = {
        'Instance': instances[instance_id],  # Thêm tên instance
        'Variables': variables_length,
        'Clauses': clauses_length,
        'Runtime': timeit.default_timer() - start,
        'Optimal_Bins': current_bins,
        'Status': 'TIMEOUT'
    }
    
    with open(f'results_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)  # Termination signal
signal.signal(signal.SIGINT, handle_interrupt)   # Keyboard interrupt (Ctrl+C)

# Create CSP_MS_R_SB folder if it doesn't exist
if not os.path.exists('CSP_MS_R_SB'):
    os.makedirs('CSP_MS_R_SB')

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
        f"inputs/CHL_format/{instance_name}.txt"
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

# Updated instance list with actual available instances for CSP
instances = [
    "",
    # BENG instances (10 instances)
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

def first_fit_upper_bound(rectangles, W, H):
    """Finite First-Fit (FFF) upper bound for 2D bin packing with rotation (Berkey & Wang)."""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    bins = []
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        # For simplicity, try to place at (0, y) for all y up to H-h
        # and check for overlap with all placed rectangles
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
        placed = False
        for bin_rects in bins:
            # Try both orientations in this bin
            for (rw, rh) in [(rect[0], rect[1]), (rect[1], rect[0])]:
                pos = fits(bin_rects, rw, rh, W, H)
                if pos is not None:
                    bin_rects.append((pos[0], pos[1], rw, rh))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # Start a new bin, place at (0,0) in best orientation
            if rect[0] <= W and rect[1] <= H:
                bins.append([(0, 0, rect[0], rect[1])])
            elif rect[1] <= W and rect[0] <= H:
                bins.append([(0, 0, rect[1], rect[0])])
            else:
                # Infeasible rectangle
                return float('inf')
    return len(bins)

def display_solution(bin_width, bin_height, rectangles, bins_assignment, positions, rotations, instance_name):
    num_bins = len(bins_assignment)
    
    if num_bins == 0:
        return

    ncols = min(num_bins, 5)
    nrows = (num_bins + ncols - 1) // ncols
    
    # Create subplots for each bin
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    fig.suptitle(f'CSP_MS_R_SB - {instance_name} - {num_bins} bins', fontsize=16)
    
    # Handle different subplot configurations
    if num_bins == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for bin_idx, items_in_bin in enumerate(bins_assignment):
        ax = axes[bin_idx]
        ax.set_title(f'Bin {bin_idx + 1}')
        
        # Draw rectangles in this bin
        for item_idx in items_in_bin:
            # Get dimensions based on rotation
            if rotations[item_idx]:
                w, h = rectangles[item_idx][1], rectangles[item_idx][0]  # Rotated
            else:
                w, h = rectangles[item_idx][0], rectangles[item_idx][1]  # Normal
            
            rect = plt.Rectangle(positions[item_idx], w, h, 
                               edgecolor="#333", facecolor="lightblue", alpha=0.6)
            ax.add_patch(rect)
            
            # Add item label with rotation indicator
            label = f"{item_idx + 1}"
            if rotations[item_idx]:
                label += "R"  # Indicate rotation
            ax.text(positions[item_idx][0] + w/2,
                   positions[item_idx][1] + h/2,
                   label, ha='center', va='center')
        
        ax.set_xlim(0, bin_width)
        ax.set_ylim(0, bin_height)
        ax.set_xticks(range(0, bin_width, max(2, bin_width // 10)))
        ax.set_yticks(range(0, bin_height, max(2, bin_height // 10)))
        ax.set_xlabel('width')
        ax.set_ylabel('height')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(num_bins, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot to CSP_MS_R_SB folder
    plt.savefig(f'CSP_MS_R_SB/{instance_name}.png')
    print(f"Solution for {instance_name} saved to CSP_MS_R_SB/{instance_name}.png")
    plt.close()

def positive_range(end):
    if end < 0:
        return []
    return range(end)

# Thêm hàm save_checkpoint để lưu tiến trình giải
def save_checkpoint(instance_id, variables, clauses, num_bins, status="IN_PROGRESS"):
    checkpoint = {
        'Variables': variables,
        'Clauses': clauses,
        'Runtime': timeit.default_timer() - start,
        'Optimal_Bins': num_bins if num_bins != float('inf') else upper_bound,
        'Status': status
    }
    
    # Ghi ra file checkpoint
    with open(f'checkpoint_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)


def CSP_MaxSAT_Rotation(rectangles, max_bins, bin_width, bin_height):
    """Solve 2D Bin Packing with given number of bins and rotation"""
    global variables_length, clauses_length, best_num_bins, best_rotations
    global optimal_solution, optimal_rotations, upper_bound, wcnf_file
    cnf = CNF()
    variables = {}
    counter = 1
    print(max_bins)

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.wcnf') as file:
        wcnf_file = file.name
        file.write(f"c 2D Bin Packing Problem with Rotation, SB symmetry, bin={bin_width}x{bin_height}, n={len(rectangles)}\n")
        file.write(f"c Lower bound={lower_bound}, Upper bound={upper_bound}\n")
    
        print(f"Creating variables for {len(rectangles)} rectangles and {max_bins} bins...")
    # Create assignment variables: x[i,j] = item i assigned to bin j
        for i in range(len(rectangles)):
            for j in range(max_bins):
                variables[f"x{i + 1},{j + 1}"] = counter
                counter += 1

        # Create position variables for each item
        for i in range(len(rectangles)):
            # Position variables for x-coordinate
            for e in range(bin_width):
                variables[f"px{i + 1},{e}"] = counter
                counter += 1
            # Position variables for y-coordinate  
            for f in range(bin_height):
                variables[f"py{i + 1},{f}"] = counter
                counter += 1

        # Create relative position variables for non-overlapping constraints
        for i in range(len(rectangles)):
            for j in range(len(rectangles)):
                if i != j:
                    variables[f"lr{i + 1},{j + 1}"] = counter  # i is left of j
                    counter += 1
                    variables[f"ud{i + 1},{j + 1}"] = counter  # i is below j
                    counter += 1

        # Rotation variables
        for i in range(len(rectangles)):
            variables[f"r{i + 1}"] = counter
            counter += 1

        # Bin usage variables
        for j in range(max_bins):
            variables[f"b{j + 1}"] = counter
            counter += 1
        cnf = []
        # Constraint 1: Each item must be assigned to exactly one bin
        for i in range(len(rectangles)):
            # At least one bin
            cnf.append([variables[f"x{i + 1},{j + 1}"] for j in range(max_bins)])
            # At most one bin
            for j1 in range(max_bins):
                for j2 in range(j1 + 1, max_bins):
                    cnf.append([-variables[f"x{i + 1},{j1 + 1}"], -variables[f"x{i + 1},{j2 + 1}"]])

        # Constraint 2: Order constraints for position variables
        for i in range(len(rectangles)):
            # x-coordinate order: px[i,e] → px[i,e+1]
            for e in range(bin_width - 1):
                cnf.append([-variables[f"px{i + 1},{e}"], variables[f"px{i + 1},{e + 1}"]])
            # y-coordinate order: py[i,f] → py[i,f+1]
            for f in range(bin_height - 1):
                cnf.append([-variables[f"py{i + 1},{f}"], variables[f"py{i + 1},{f + 1}"]])

        # Constraint 3: Bin usage constraints
        for j in range(max_bins):
            for i in range(len(rectangles)):
                # If item i is in bin j, then bin j is used
                cnf.append([-variables[f"x{i + 1},{j + 1}"], variables[f"b{j + 1}"]])

        # Constraint 4: Symmetry Breaking - bin ordering
        for j in range(1, max_bins):
            cnf.append([-variables[f"b{j + 1}"], variables[f"b{j}"]])
        
        print(f"Number of clauses before symmetry breaking: {len(cnf)}")

        # Constraint 5: Non-overlapping constraints with rotation
        def add_non_overlapping(rotated, i, j, bin_idx, h1, h2, v1, v2):
            """Add non-overlapping constraints for items i and j in bin bin_idx with rotation"""
            
            # Get dimensions based on rotation
            if not rotated:
                i_width = rectangles[i][0]
                i_height = rectangles[i][1]
                j_width = rectangles[j][0]
                j_height = rectangles[j][1]
                i_rotation = variables[f"r{i + 1}"]
                j_rotation = variables[f"r{j + 1}"]
            else:
                i_width = rectangles[i][1]
                i_height = rectangles[i][0]
                j_width = rectangles[j][1]
                j_height = rectangles[j][0]
                i_rotation = -variables[f"r{i + 1}"]
                j_rotation = -variables[f"r{j + 1}"]
            
            bin_condition = [-variables[f"x{i + 1},{bin_idx + 1}"], -variables[f"x{j + 1},{bin_idx + 1}"]]
            
            # Four-literal clause with rotation conditions
            four_literal = bin_condition.copy()
            if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
            if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
            if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
            if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])
            
            cnf.append(four_literal + [i_rotation])
            cnf.append(four_literal + [j_rotation])

            # Position-based constraints with rotation
            if h1:
                for e in range(min(bin_width, i_width)):
                    cnf.append([i_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"lr{i + 1},{j + 1}"], -variables[f"px{j + 1},{e}"]])
            
            if h2:
                for e in range(min(bin_width, j_width)):
                    cnf.append([j_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"lr{j + 1},{i + 1}"], -variables[f"px{i + 1},{e}"]])

            if v1:
                for f in range(min(bin_height, i_height)):
                    cnf.append([i_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"ud{i + 1},{j + 1}"], -variables[f"py{j + 1},{f}"]])
            
            if v2:
                for f in range(min(bin_height, j_height)):
                    cnf.append([j_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"ud{j + 1},{i + 1}"], -variables[f"py{i + 1},{f}"]])

            # Position-based non-overlapping with rotation
            for e in positive_range(bin_width - i_width):
                if h1:
                    cnf.append([i_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"lr{i + 1},{j + 1}"],
                            variables[f"px{i + 1},{e}"],
                            -variables[f"px{j + 1},{e + i_width}"]])

            for e in positive_range(bin_width - j_width):
                if h2:
                    cnf.append([j_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"lr{j + 1},{i + 1}"],
                            variables[f"px{j + 1},{e}"],
                            -variables[f"px{i + 1},{e + j_width}"]])

            for f in positive_range(bin_height - i_height):
                if v1:
                    cnf.append([i_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"ud{i + 1},{j + 1}"],
                            variables[f"py{i + 1},{f}"],
                            -variables[f"py{j + 1},{f + i_height}"]])
            
            for f in positive_range(bin_height - j_height):
                if v2:
                    cnf.append([j_rotation, bin_condition[0], bin_condition[1],
                            -variables[f"ud{j + 1},{i + 1}"],
                            variables[f"py{j + 1},{f}"],
                            -variables[f"py{i + 1},{f + j_height}"]])

        # Find maximum width for symmetry breaking
        max_width = max([int(rectangle[0]) for rectangle in rectangles])
        second_max_width = max([int(rectangle[0]) for rectangle in rectangles if int(rectangle[0]) != max_width])

        # Apply non-overlapping constraints for all pairs in all bins with symmetry breaking
        for bin_idx in range(max_bins):
            for i in range(len(rectangles)):
                for j in range(i + 1, len(rectangles)):
                    # Symmetry Breaking similar to SPP_R_SB
                    if rectangles[i][0] == max_width and rectangles[j][0] == second_max_width:
                        # Fix the position of the largest and second largest rectangle
                        add_non_overlapping(False, i, j, bin_idx, False, False, True, True)
                        add_non_overlapping(True, i, j, bin_idx, False, False, True, True)
                    elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > bin_width:
                        # Large-rectangles horizontal
                        add_non_overlapping(False, i, j, bin_idx, False, False, True, True)
                        add_non_overlapping(True, i, j, bin_idx, False, False, True, True)
                    elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > bin_height:
                        # Large rectangles vertical
                        add_non_overlapping(False, i, j, bin_idx, True, True, False, False)
                        add_non_overlapping(True, i, j, bin_idx, True, True, False, False)
                    else:
                        # Normal rectangles
                        add_non_overlapping(False, i, j, bin_idx, True, True, True, True)
                        add_non_overlapping(True, i, j, bin_idx, True, True, True, True)

        # Constraint 6: Domain constraints - items must fit within bins
        for i in range(len(rectangles)):
            for bin_idx in range(max_bins):
                # Normal orientation
                if rectangles[i][0] > bin_width:
                    cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], variables[f"r{i + 1}"]])
                else:
                    for e in range(bin_width - rectangles[i][0], bin_width):
                        cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], variables[f"r{i + 1}"],
                                variables[f"px{i + 1},{e}"]])
                
                if rectangles[i][1] > bin_height:
                    cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], variables[f"r{i + 1}"]])
                else:
                    for f in range(bin_height - rectangles[i][1], bin_height):
                        cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{f}"]])

                # Rotated orientation
                if rectangles[i][1] > bin_width:
                    cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], -variables[f"r{i + 1}"]])
                else:
                    for e in range(bin_width - rectangles[i][1], bin_width):
                        cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], -variables[f"r{i + 1}"],
                                variables[f"px{i + 1},{e}"]])
                
                if rectangles[i][0] > bin_height:
                    cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], -variables[f"r{i + 1}"]])
                else:
                    for f in range(bin_height - rectangles[i][0], bin_height):
                        cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], -variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{f}"]])
                        
        soft_clauses = []
        # Soft clauses: minimize the number of bins used
        for j in range(lower_bound, max_bins):
            soft_clauses.append((1, [-variables[f"b{j + 1}"]]))
        # Write hard clauses with 'h' prefix
        for clause in cnf:
            file.write(f"h {' '.join(map(str, clause))} 0\n")
        
        # Write soft clauses with their weights
        for weight, clause in soft_clauses:
            file.write(f"{weight} {' '.join(map(str, clause))} 0\n")
        
        # For debugging, print details about the WCNF file
        print(f"Created WCNF file with: {len(cnf)} hard clauses and {len(soft_clauses)} soft clauses")
        print(f"Variable count: {counter-1}")
        file.flush()


    variables_length = len(variables)
    clauses_length = len(cnf) + len(soft_clauses)
    
    # Save checkpoint
    save_checkpoint(instance_id, variables_length, clauses_length, best_num_bins)

    try:
        print(f"Running tt-open-wbo-inc on {wcnf_file}...")
        result = subprocess.run(
            ["./tt-open-wbo-inc-Glucose4_1_static", wcnf_file], 
            capture_output=True, 
            text=True
        )
        
        output = result.stdout
        print(f"Solver output preview: {output[:200]}...")  # Debug: Show beginning of output
        
        # Parse the output to find the model
        optimal_bins = max_bins
        bins_assignment = [[] for _ in range(max_bins)]
        positions = [[0, 0] for _ in range(len(rectangles))]
        rotations = [False for _ in range(len(rectangles))]
        
        if "OPTIMUM FOUND" in output:
            print("Optimal solution found!")
            
            # Extract the model line (starts with "v ")
            for line in output.split('\n'):
                if line.startswith('v '):
                    print(f"Found solution line: {line[:50]}...")  # Debug output
                    
                    # New format: v 01010101010... (continuous binary string)
                    # Remove the 'v ' prefix
                    binary_string = line[2:].strip()
                    
                    # Diagnostic information
                    print("\nSOLVER OUTPUT DIAGNOSTICS:")
                    print("=" * 50)
                    print(f"Characters in solution: {set(binary_string)}")
                    print(f"First 20 characters: {binary_string[:20]}")
                    print(f"Length of binary string: {len(binary_string)}")
                    print("=" * 50)
                    
                    # Convert binary values to true variable set
                    true_vars = set()
                    
                    # Process the solution string - tt-open-wbo-inc can output in different formats
                    # Try to interpret as space-separated list of integers first
                    if " " in binary_string:
                        try:
                            for val in binary_string.split():
                                val_int = int(val)
                                if val_int > 0:  # Positive literals represent true variables
                                    true_vars.add(val_int)
                        except ValueError:
                            # Not integers, try as space-separated binary values
                            for i, val in enumerate(binary_string.split()):
                                if val == '1':
                                    true_vars.add(i + 1)  # 1-indexed
                    else:
                        # No spaces - treat as continuous binary string
                        for i, val in enumerate(binary_string):
                            if val == '1':
                                true_vars.add(i + 1)
                    print(f"Found {len(true_vars)} true variables out of {len(binary_string)} total")
                    bins_assignment = [[] for _ in range(max_bins)]
                    # Extract bin usage variables and find minimum bins where pb_b is true
                    pb_true_bins = []
                    for b in range(1, max_bins + 1):
                        var_key = f"b{b}"
                        if var_key in variables and variables[var_key] in true_vars:
                            pb_true_bins.append(b)
                    print(f"b variables: {pb_true_bins}")
                    if pb_true_bins:
                        optimal_bins = len(pb_true_bins)
                        print(f"Optimal number of bins: {optimal_bins}")                        
                        # Update best number of bins if better solution found
                        if optimal_bins < best_num_bins:
                            best_num_bins = optimal_bins
                            save_checkpoint(instance_id, variables_length, clauses_length, best_num_bins)
                    else:
                        print("WARNING: No b variables are true! This may indicate a parsing issue.")
                        # Use lower bound as fallback
                        optimal_bins = lower_bound
                        best_num_bins = lower_bound
                    # If we couldn't parse any variables but the solver found a solution,
                    # use the lower bound as a fallback
                    if not true_vars:
                        print("WARNING: Solution parsing failed. Using lower bound bins as fallback.")
                        optimal_bins = lower_bound
                        best_num_bins = lower_bound
                        
                    else:
                        # Extract rotation variables
                        for i in range(len(rectangles)):
                            if variables[f"r{i + 1}"] in true_vars:
                                rotations[i] = True
                        
                        # Extract bin assignments
                        bins_assignment = [[] for _ in range(optimal_bins)]
                        for i in range(len(rectangles)):
                            for b in range(optimal_bins):
                                var_key = f"x{i + 1},{b + 1}"
                                if var_key in variables and variables[var_key] in true_vars:
                                    bins_assignment[b].append(i)
                                    break
                        
                        # Extract positions
                        for i in range(len(rectangles)):
                            # Find x position (first position where px is true)
                            found_x = False
                            for e in range(bin_width):
                                var_key = f"px{i + 1},{e}"
                                if var_key in variables and variables[var_key] in true_vars:
                                    if e == 0 or variables[f"px{i + 1},{e-1}"] not in true_vars:
                                        positions[i][0] = e
                                        found_x = True
                                        break
                            if not found_x:
                                print(f"WARNING: Could not determine x-position for rectangle {i}!")
                            
                            # Find y position (first position where py is true)
                            found_y = False
                            for f in range(bin_height):
                                var_key = f"py{i + 1},{f}"
                                if var_key in variables and variables[var_key] in true_vars:
                                    if f == 0 or variables[f"py{i + 1},{f-1}"] not in true_vars:
                                        positions[i][1] = f
                                        found_y = True
                                        break
                            if not found_y:
                                print(f"WARNING: Could not determine y-position for rectangle {i}!")
                    # Save the results
                    bins_assignment = [bin_items for bin_items in bins_assignment if bin_items]
                    best_rotations = rotations
                    optimal_solution = positions
                    optimal_rotations = rotations
                    break
        else:
            print("No optimal solution found.")
            print(f"Solver output: {output}")
            return None, None, None, None
        # Clean up the temporary file
        os.remove(wcnf_file)
        return optimal_bins, bins_assignment, positions, rotations
    except Exception as e:
        print(f"Error running Max-SAT solver: {e}")
        traceback.print_exc()
        if os.path.exists(wcnf_file):
            os.remove(wcnf_file)
        return None, None, None, None


if __name__ == "__main__":
    # Phần controller mode
    if len(sys.argv) == 1:
        # This is the controller mode - running without arguments
        # Create CSP_MS_R_SB folder if it doesn't exist
        if not os.path.exists('CSP_MS_R_SB'):
            os.makedirs('CSP_MS_R_SB')
        
        # Đọc file Excel hiện có để kiểm tra instances đã hoàn thành
        excel_file = 'CSP_MS_R_SB.xlsx'
        if os.path.exists(excel_file):
            # Đọc file Excel hiện có nếu nó tồn tại
            existing_df = pd.read_excel(excel_file)
            # Lấy danh sách các instance đã hoàn thành
            completed_instances = existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else []
        else:
            # Tạo DataFrame trống nếu chưa có file
            existing_df = pd.DataFrame()
            completed_instances = []
        
        # Set timeout in seconds
        TIMEOUT = 900  # 15 minutes timeout

        for instance_id in range(1, len(instances)):
            instance_name = instances[instance_id]
            
            # Kiểm tra xem instance này đã được chạy chưa
            if instance_name in completed_instances:
                print(f"\nSkipping instance {instance_id}: {instance_name} (already completed)")
                continue
                
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name}")
            print(f"{'=' * 50}")
            
            # Clean up any previous result file
            if os.path.exists(f'results_{instance_id}.json'):
                os.remove(f'results_{instance_id}.json')
            
            # Run the instance with runlim, but use THIS script with the instance_id
            command = f"./runlim -r {TIMEOUT} python3 CSP_MS_R_SB.py {instance_id}"
            
            try:
                # Run the command and wait for it to complete
                process = subprocess.Popen(command, shell=True)
                process.wait()
                
                # Wait a moment to ensure file is written
                time.sleep(1)
                
                # Kiểm tra kết quả
                result = None
                
                # Thử đọc file results trước (kết quả hoàn chỉnh)
                if os.path.exists(f'results_{instance_id}.json'):
                    with open(f'results_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                
                # Nếu không tìm thấy file results, kiểm tra file checkpoint
                elif os.path.exists(f'checkpoint_{instance_id}.json'):
                    with open(f'checkpoint_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                    # Đánh dấu đây là kết quả timeout
                    result['Status'] = 'TIMEOUT'
                    result['Instance'] = instance_name
                    print(f"Instance {instance_name} timed out. Using checkpoint data.")
                
                # Xử lý kết quả (nếu có)
                if result:
                    print(f"Instance {instance_name} - Status: {result['Status']}")
                    print(f"Optimal Bins: {result['Optimal_Bins']}, Runtime: {result['Runtime']}")
                    
                    # Cập nhật Excel
                    if os.path.exists(excel_file):
                        try:
                            existing_df = pd.read_excel(excel_file)
                            instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                            
                            if instance_exists:
                                # Cập nhật instance đã tồn tại
                                instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                                for key, value in result.items():
                                    existing_df.at[instance_idx, key] = value
                            else:
                                # Thêm instance mới
                                result_df = pd.DataFrame([result])
                                existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                        except Exception as e:
                            print(f"Lỗi khi đọc file Excel hiện có: {str(e)}")
                            existing_df = pd.DataFrame([result])
                    else:
                        # Tạo DataFrame mới nếu chưa có file Excel
                        existing_df = pd.DataFrame([result])
                    # Lưu DataFrame vào Excel
                    existing_df.to_excel(excel_file, index=False)
                    print(f"Results saved to {excel_file}")
                        
                else:
                    print(f"No results or checkpoint found for instance {instance_name}")
                    
            except Exception as e:
                print(f"Error running instance {instance_name}: {str(e)}")
            import glob
            temp_wcnf_files = glob.glob('/tmp/tmp*.wcnf')
            for temp_file in temp_wcnf_files:
                try:
                    # Only remove files that are older than 1 minute to avoid removing active files
                    if os.path.exists(temp_file):
                        file_age = time.time() - os.path.getmtime(temp_file)
                        if file_age >= 900:  
                            os.remove(temp_file)
                            print(f"Cleaned up temporary file: {temp_file}")
                except Exception as cleanup_error:
                    # Silently ignore cleanup errors
                    pass
            # Clean up the results file to avoid confusion
            for file in [f'results_{instance_id}.json', f'checkpoint_{instance_id}.json']:
                if os.path.exists(file):
                    os.remove(file)
        
        print(f"\nAll instances completed. Results saved to {excel_file}")
    
    # Phần single instance mode
    else:
        # Single instance mode
        instance_id = int(sys.argv[1])
        instance_name = instances[instance_id]
        
        start = timeit.default_timer()  # start clock
        
        try:
            print(f"\nProcessing instance {instance_name}")
            
            # Reset global best solution for this instance
            best_num_bins = float('inf')
            best_positions = []
            best_rotations = []

            # read file input
            input = read_file_instance(instance_name)
            n_rec = int(input[0])
            
            # Parse bin dimensions from line 2 (format: "width height")
            bin_dimensions = input[1].split()
            bin_width = int(bin_dimensions[0])
            bin_height = int(bin_dimensions[1])
            
            rectangles = []
            for line in input[2:2 + n_rec]:
                demand = int(line.split()[2])
                for i in range(demand):
                    rectangles.append([int(line.split()[0]), int(line.split()[1])])
            
            # Calculate initial bounds
            # For rotation, we need to consider both orientations for bins calculation
            total_area = sum([w * h for w, h in rectangles])
            bin_area = bin_width * bin_height
            
            # Lower bound: total area divided by bin area (ceiling)
            lower_bound = math.ceil(total_area / bin_area)
            
            # Upper bound: use first fit with rotation
            upper_bound = first_fit_upper_bound(rectangles, bin_width, bin_height)
            
            # Ensure lower bound is not greater than upper bound
            if lower_bound > upper_bound:
                print(f"Warning: Lower bound ({lower_bound}) > Upper bound ({upper_bound}). Adjusting...")
                lower_bound = upper_bound

            print(f"Solving 2D Bin Packing with MaxSAT (with rotation, SB symmetry) for instance {instance_name}")
            print(f"Bin dimensions: {bin_width} x {bin_height}")
            print(f"Number of rectangles: {n_rec}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            
            # Solve with MaxSAT
            optimal_bins, optimal_assignment, optimal_pos, optimal_rot = CSP_MaxSAT_Rotation(rectangles, upper_bound, bin_width, bin_height)
            
            stop = timeit.default_timer()
            runtime = stop - start

            # Display and save the solution if we found one
            if optimal_bins is not None and optimal_assignment is not None and optimal_pos is not None and optimal_rot is not None:
                display_solution(bin_width, bin_height, rectangles, optimal_assignment, optimal_pos, optimal_rot, instance_name)

            # Tạo result object
            result = {
                'Instance': instance_name,
                'Variables': variables_length,
                'Clauses': clauses_length,
                'Runtime': runtime,
                'Optimal_Bins': optimal_bins if optimal_bins is not None else upper_bound,
                'Status': 'COMPLETE' if optimal_bins is not None else 'ERROR'
            }
            
            # Ghi kết quả vào Excel trực tiếp
            excel_file = 'CSP_MS_R_SB.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        # Cập nhật instance đã tồn tại
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        # Thêm instance mới
                        result_df = pd.DataFrame([result])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except Exception as e:
                    print(f"Lỗi khi đọc file Excel hiện có: {str(e)}")
                    existing_df = pd.DataFrame([result])
            else:
                # Tạo DataFrame mới nếu chưa có file Excel
                existing_df = pd.DataFrame([result])
            
            # Lưu DataFrame vào Excel
            existing_df.to_excel(excel_file, index=False)
            print(f"Results saved to {excel_file}")
            
            # Save result to a JSON file that the controller will read
            with open(f'results_{instance_id}.json', 'w') as f:
                json.dump(result, f)
            
            print(f"Instance {instance_name} completed - Runtime: {runtime:.2f}s, Bins: {optimal_bins}")

        except Exception as e:
            print(f"Error in instance {instance_name}: {str(e)}")
            traceback.print_exc()  # Print the traceback for the error
            
            # Save error result - use upper_bound if no best_num_bins
            current_bins = best_num_bins if best_num_bins != float('inf') else upper_bound
            result = {
                'Instance': instance_name,
                'Variables': variables_length,
                'Clauses': clauses_length,
                'Runtime': timeit.default_timer() - start,
                'Optimal_Bins': current_bins,
                'Status': 'ERROR'
            }
            
            # Ghi kết quả lỗi vào Excel
            excel_file = 'CSP_MS_R_SB.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        # Cập nhật instance đã tồn tại
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        # Thêm instance mới
                        result_df = pd.DataFrame([result])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except Exception as ex:
                    print(f"Lỗi khi đọc file Excel hiện có: {str(ex)}")
                    existing_df = pd.DataFrame([result])
            else:
                # Tạo DataFrame mới nếu chưa có file Excel
                existing_df = pd.DataFrame([result])
            
            # Lưu DataFrame vào Excel
            existing_df.to_excel(excel_file, index=False)
            print(f"Error results saved to {excel_file}")
            
            # Save result to a JSON file that the controller will read
            with open(f'results_{instance_id}.json', 'w') as f:
                json.dump(result, f)
        