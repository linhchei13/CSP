def convert_format(input_file, output_file):
    """
    Convert CHL format to new format:
    From: W H \n n_items \n w h d value
    To: n_items \n W H \n w h d
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse input
    W, H = map(int, lines[0].strip().split())
    n_items = int(lines[1].strip())
    
    items = []
    for i in range(2, 2 + n_items):
        parts = lines[i].strip().split()
        w, h, d = int(parts[0]), int(parts[1]), int(parts[2])
        items.append((w, h, d))
    
    # Write new format
    with open(output_file, 'w') as f:
        f.write(f"{n_items}\n")
        f.write(f"{W} {H}\n")
        for w, h, d in items:
            f.write(f"{w} {h} {d}\n")

def convert_beng_dataset(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse input
    n_items = int(lines[0].strip())
    W, H = map(int, lines[1].strip().split())
    
    items = []
    for i in range(2, 2 + n_items):
        parts = lines[i].strip().split()
        w, h, d = int(parts[1]), int(parts[2]), int(parts[3])
        items.append((w, h, d))
    
    # Write new format
    with open(output_file, 'w') as f:
        f.write(f"{n_items}\n")
        f.write(f"{W} {H}\n")
        for w, h, d in items:
            f.write(f"{w} {h} {d}\n")
# Convert CHL1 file
# for i in range(2, 10):
convert_beng_dataset(f'inputs/BENG/BENG10.ins2D', f'inputs/BENG/BENG10.txt')
print("Conversion completed!")