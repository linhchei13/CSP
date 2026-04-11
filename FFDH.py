import math
import os

import pandas as pd


def list_instance_files():
    """Return all instance file paths from known input folders."""
    folders = [
        "inputs/BENG",
        "inputs/set1",
        "inputs/set2",
        "inputs/set3",
    ]
    files = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.endswith(".txt"):
                files.append(os.path.join(folder, name))
    return sorted(files)


def read_instance(file_path):
    """Read one instance file and return (instance_name, W, H, rectangles)."""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n_items = int(lines[0])
    w_h = lines[1].split()
    bin_width, bin_height = int(w_h[0]), int(w_h[1])

    rectangles = []
    for line in lines[2:2 + n_items]:
        parts = line.split()
        w = int(parts[0])
        h = int(parts[1])
        demand = int(parts[2]) if len(parts) >= 3 else 1
        for _ in range(demand):
            rectangles.append([w, h])

    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    return instance_name, bin_width, bin_height, rectangles


def first_fit_upper_bound_with_rotation(rectangles, W, H):
    """First-fit heuristic to get upper bound with rotation allowed"""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    sorted_rectangles = rectangles.copy()  # Avoid modifying original list
    sorted_rectangles.sort(key=lambda x: max(x[0], x[1]), reverse=True)  # Sort by max dimension for better packing
    bins = []
    
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                overlap = False
                for (px, py, pw, ph) in bin_rects:
                    if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
        return None
    
    for rect in sorted_rectangles:
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

def first_fit_decreasing_height(items, bin_width, bin_height):
    # Sort items by height in decreasing order
    items.sort(key=lambda x: x[1], reverse=True)
    
    bins = []
    
    for item in items:
        placed = False
        for b in bins:
            if sum(i[0] for i in b) + item[0] <= bin_width:
                b.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])
    
    return bins


def first_fit_upper_bound_no_rotation(rectangles, W, H):
    """First-fit heuristic to get upper bound without rotation."""
    sorted_rectangles = sorted(rectangles, key=lambda x: (x[1], x[0]), reverse=True)
    bins = []

    def fits(bin_rects, w, h, W, H):
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                overlap = False
                for (px, py, pw, ph) in bin_rects:
                    if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
        return None

    for rect in sorted_rectangles:
        w, h = rect[0], rect[1]
        if w > W or h > H:
            return float('inf')

        placed = False
        for bin_rects in bins:
            pos = fits(bin_rects, w, h, W, H)
            if pos is not None:
                bin_rects.append((pos[0], pos[1], w, h))
                placed = True
                break

        if not placed:
            bins.append([(0, 0, w, h)])

    return len(bins)


def first_fit_upper_bound(rectangles, W, H):
    """Fast shelf-based FFDH upper bound for 2D bin packing with rotation."""
    # Sort rectangles by decreasing max dimension for better packing
    sorted_rects = sorted(rectangles, key=lambda r: max(r[0], r[1]), reverse=True)
    # Each bin has shelves with (height, remaining_width)
    bins = []

    for rect in sorted_rects:
        w, h = rect[0], rect[1]
        # Build valid orientations, prefer shorter height first (fits more shelves)
        orientations = []
        if w <= W and h <= H:
            orientations.append((w, h))
        if h <= W and w <= H and w != h:
            orientations.append((h, w))
        orientations.sort(key=lambda o: o[1])

        if not orientations:
            return float('inf')

        placed = False
        # Try to fit in existing bins' existing shelves (First Fit)
        for bin_info in bins:
            for ow, oh in orientations:
                for shelf in bin_info['shelves']:
                    if shelf['remaining_w'] >= ow and shelf['height'] >= oh:
                        shelf['remaining_w'] -= ow
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

        if not placed:
            # Try to create a new shelf in existing bins
            for bin_info in bins:
                for ow, oh in orientations:
                    if bin_info['used_height'] + oh <= H and ow <= W:
                        bin_info['shelves'].append({'height': oh, 'remaining_w': W - ow})
                        bin_info['used_height'] += oh
                        placed = True
                        break
                if placed:
                    break

        if not placed:
            # Create a new bin with a new shelf
            ow, oh = orientations[0]
            bins.append({
                'shelves': [{'height': oh, 'remaining_w': W - ow}],
                'used_height': oh
            })

    return len(bins)


def compute_upper_bounds_for_all_instances(output_file="FFDH.xlsx"):
    results = []
    instance_files = list_instance_files()

    for file_path in instance_files:
        instance_name, bin_width, bin_height, rectangles = read_instance(file_path)

        ub_first_fit_rotation = first_fit_upper_bound_with_rotation(rectangles, bin_width, bin_height)
        ub_ffdh = len(first_fit_decreasing_height(rectangles.copy(), bin_width, bin_height))
        ub_first_fit = first_fit_upper_bound_no_rotation(rectangles, bin_width, bin_height)

        results.append({
            "Instance": instance_name,
            "Source_File": file_path,
            "Bin_Width": bin_width,
            "Bin_Height": bin_height,
            "Num_Rectangles": len(rectangles),
            "First_Fit_Rotation_UB": ub_first_fit_rotation,
            "FFDH_UB": ub_ffdh,
            "First_Fit_UB": ub_first_fit,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["Source_File", "Instance"]).reset_index(drop=True)
    df.to_excel(output_file, index=False)
    print(f"Saved upper bounds for {len(df)} instances to {output_file}")


if __name__ == "__main__":
    compute_upper_bounds_for_all_instances()

