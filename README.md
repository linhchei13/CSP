# Giải bài toán Cắt vật liệu hai chiều bằng SAT, Incremental SAT và MaxSAT.

Repository này chứa mã nguồn của tất cả các solver được đánh giá trong khoá luận, cùng với bộ dữ liệu  được sử dụng trong thực nghiệm.

---

## Mô tả Bài toán

**Bài toán Cắt vật liệu hai chiều (Two-Dimensional Single Stock Size Cutting Stock Problem - 2D-CSSP)**

Đầu vào:
- Một tập hợp n loại hình chữ nhật, mỗi loại có chiều rộng w_i,  chiều cao h_i và nhu cầu (số lượng bản sao cần thiết) d_i.
- Các tấm vật liệu thô có kích thước giống nhau

**Mục tiêu:** **Tối thiểu hóa số lượng tấm sử dụng**.

---

## 🔧 Các Cấu hình Solver

Mỗi script Python triển khai một cách tiếp cận giải quyết cụ thể. Quy ước đặt tên:

| Hậu tố | Ý nghĩa |
|--------|---------|
| *(không có)* | Không xoay, không phá vỡ đối xứng |
| `_R` | Cho phép xoay 90° |
| `_SB` | Phá vỡ đối xứng |
| `_R_SB` | Cả xoay lẫn phá vỡ đối xứng |

### Phương pháp dựa trên SAT (PySAT / Glucose 4.2)

| Script | Phương pháp |
|--------|---------|
| `CSP.py` | Chạy SAT nhiều lần + không xoay + không phá vỡ đối xứng |
| `CSP_R.py` | Chạy SAT nhiều lần + xoay + không phá vỡ đối xứng|
| `CSP_SB.py` | Chạy SAT nhiều lần + không xoay + phá vỡ đối xứng |
| `CSP_R_SB.py` | Chạy SAT nhiều lần + xoay + phá vỡ đối xứng |
| `CSP_INC.py` | SAT tăng dần + không xoay + không phá vỡ đối xứng |
| `CSP_INC_R.py` | SAT tăng dần + xoay + không phá vỡ đối xứng |
| `CSP_INC_SB.py` | SAT tăng dần + không xoay + phá vỡ đối xứng |
| `CSP_INC_R_SB.py` | SAT tăng dần + xoay + phá vỡ đối xứng |

### 🎯 MaxSAT (TT-Open-WBO-Inc)

| Script | Phương pháp |
|--------|---------|
| `CSP_MS.py` | MaxSAT + không xoay + không phá vỡ đối xứng |
| `CSP_MS_R.py` | MaxSAT + xoay + không phá vỡ đối xứng| |
| `CSP_MS_SB.py` | MaxSAT + không xoay + phá vỡ đối xứng |
| `CSP_MS_R_SB.py` | MaxSAT + xoay + phá vỡ đối xứng |

### 💼 Bộ giải thương mại (CPLEX / Gurobi / OR-Tools)

| Script | Phương pháp |
|--------|---------|
| `CPLEX_CP_SB.py` | CPLEX CP + không xoay + phá vỡ đối xứng |
| `CPLEX_CP_R_SB.py` | CPLEX CP + xoay + phá vỡ đối xứng |
| `CPLEX_MIP_SB.py` | CPLEX MIP + không xoay + phá vỡ đối xứng |
| `CPLEX_MIP_R_SB.py` | CPLEX MIP + xoay + phá vỡ đối xứng |
| `GUROBI_MIP_SB.py` | Gurobi MIP + không xoay +  phá vỡ đối xứng |
| `GUROBI_MIP_R_SB.py` | Gurobi MIP + xoay + phá vỡ đối xứng |
| `OR-TOOLS_CP_SB.py` | OR-Tools CP-SAT + không xoay + phá vỡ đối xứng |
| `OR-TOOLS_CP_R_SB.py` | OR-Tools CP-SAT  + xoay + phá vỡ đối xứng |
| `OR-TOOLS_MIP_SB.py` | OR-Tools MIP + không xoay + phá vỡ đối xứng |
| `OR-TOOLS_MIP_R_SB.py` | OR-Tools MIP + xoay + phá vỡ đối xứng |

---

## Cài đặt

### Bước 1: Tạo và kích hoạt môi trường ảo

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 2: Cài đặt các thư viện Python

```bash
pip install -r src/requirements.txt
```

**Thư viện cần thiết:** `python-sat`, `pandas`, `matplotlib`, `numpy`, `openpyxl`, `ortools`, `gurobipy`, `cplex`, `docplex`

> ⚠️ **Ghi chú:**
> - **Gurobi** và **CPLEX** yêu cầu lincense
> - **OR-Tools** là mã nguồn mở, cài đặt via pip

### Bước 3: Cài đặt công cụ khác

Các bộ giải dựa trên SAT cần hai công cụ bên ngoài:

#### `runlim` — Giới hạn tài nguyên và thời gian

```bash
# cd src
git clone https://github.com/arminbiere/runlim.git
```

### `tt-open-wbo-inc-Glucose4_1_static` — Bộ giải MaxSAT

```bash
# cd src
# Tùy chọn 1: Clone từ GitHub
git clone https://github.com/alexander-nadel-academic/tt-open-wbo-inc.git


```

---

##  Cách chạy
Mỗi script hoạt động ở hai chế độ:

**Chế độ Controller** (chạy tất cả các instance tuần tự với giới hạn thời gian):
```bash
cd src
python3 CSP.py
```

**Chế độ Instance duy nhất** (chạy một instance theo index):
```bash
cd src
python3 CSP.py <instance_id>
```

**Ví dụ:**
```bash
python3 CSP_INC_R_SB.py 3        
python3 CPLEX_MIP_R_SB.py 5 
```

Kết quả được lưu dưới file `.xlsx`


## 📋 Định dạng Input

Mỗi file dữ liệu có cấu trúc sau:

```
<n_item_types>
<W> <H>
<w_1> <h_1> <d_1>
<w_2> <h_2> <d_2>
...
```

**Trong đó:**
- `n_item_types` — số loại hình chữ nhật khác nhau
- `W`, `H` — chiều rộng và chiều cao của tấm vật liệu
- `w_i`, `h_i`, `d_i` — chiều rộng, chiều cao và nhu cầu (số lượng bản sao) của loại hình chữ nhật thứ i

**Ví dụ:**
```
3
100 100
20 30 5
40 40 3
50 25 2
```

Điều này có nghĩa là:
- 3 loại hình chữ nhật
- Tấm vật liệu có kích thước 100×100
- Loại 1: 20×30, cần 5 bản sao
- Loại 2: 40×40, cần 3 bản sao
- Loại 3: 50×25, cần 2 bản sao

---

##  Bộ dữ liệu 

Tất cả các file instance nằm trong thư mục `inputs/set2/`:

**Bộ dữ liệu Cui–Zhao** — Bộ dữ liệu chính được sử dụng trong khoá luận với 30 instances.

---

##  Cấu trúc 

```
.
├── README.md                      # File hướng dẫn 
├── src/                           # Thư mục chính
│   ├── inputs/
│   │   └── set2/                  # Bộ dữ Cui-Zhao
│   ├── CSP.py                     # Chạy SAT nhiều lần
│   ├── CSP_R.py                   # Chạy SAT nhiều lần + xoay
│   ├── CSP_SB.py                  # Chạy SAT nhiều lần + phá vỡ đối xứng
│   ├── CSP_R_SB.py                # Chạy SAT nhiều lần + xoay + phá vỡ đối xứng
│   ├── CSP_INC.py                 # SAT tăng dần
│   ├── CSP_INC_R.py               # SAT tăng dần + xoay
│   ├── CSP_INC_SB.py              # SAT tăng dần + phá vỡ đối xứng
│   ├── CSP_INC_R_SB.py            # SAT tăng dần + xoay + phá vỡ đối xứng
│   ├── CSP_MS.py                  # MaxSAT
│   ├── CSP_MS_R.py                # MaxSAT + xoay
│   ├── CSP_MS_SB.py               # MaxSAT + phá vỡ đối xứng
│   ├── CSP_MS_R_SB.py             # MaxSAT + xoay + phá vỡ đối xứng
│   ├── CPLEX_CP_SB.py             # CPLEX CP + phá vỡ đối xứng
│   ├── CPLEX_CP_R_SB.py           # CPLEX CP + xoay + phá vỡ đối xứng
│   ├── CPLEX_MIP_SB.py            # CPLEX MIP + phá vỡ đối xứng
│   ├── CPLEX_MIP_R_SB.py          # CPLEX MIP + xoay + phá vỡ đối xứng
│   ├── GUROBI_MIP_SB.py           # Gurobi MIP + phá vỡ đối xứng
│   ├── GUROBI_MIP_R_SB.py         # Gurobi MIP + xoay + phá vỡ đối xứng
│   ├── OR-TOOLS_CP_SB.py          # OR-Tools CP-SAT + phá vỡ đối xứng
│   ├── OR-TOOLS_CP_R_SB.py        # OR-Tools CP-SAT + xoay + phá vỡ đối xứng
|   |── OR-TOOLS_CP_SB.py          # OR-Tools MIP + phá vỡ đối xứng
│   ├── OR-TOOLS_CP_R_SB.py        # OR-Tools MIP + xoay + phá vỡ đối xứng
│   ├── requirements.txt           # Các thư viện Python cần thiết
│   ├── runlim                     # Công cụ giới hạn tài nguyên
│   ├── tt-open-wbo-inc-Glucose4_1_static  # Bộ giải MaxSAT
└── references/                    # Các tài liệu tham khảo
```