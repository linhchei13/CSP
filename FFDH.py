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