from ortools.linear_solver import pywraplp
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    # Đọc dữ liệu đầu vào
    # Line 1: n, m, k, s, t
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    s = int(data[idx]); idx += 1
    t = int(data[idx]); idx += 1

    # Đọc danh sách cạnh (u, v, cost)
    edges = []  # mỗi phần tử: (u, v, cost)
    for _ in range(m):
        u = int(data[idx]); v = int(data[idx+1]); cost = int(data[idx+2])
        idx += 3
        edges.append((u, v, cost))
    
    # Đọc danh sách cặp cạnh cấm: mỗi dòng gồm 4 số: u1, v1, u2, v2
    forbidden_pairs = []
    for _ in range(k):
        u1 = int(data[idx]); v1 = int(data[idx+1])
        u2 = int(data[idx+2]); v2 = int(data[idx+3])
        idx += 4
        forbidden_pairs.append(((u1, v1), (u2, v2)))
    
    # Tạo solver với SCIP
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Không tạo được solver SCIP!")
        return

    # Tạo biến quyết định: x[u,v] = 1 nếu cạnh (u,v) được chọn, 0 nếu không
    x = {}
    for (u, v, cost) in edges:
        x[(u, v)] = solver.IntVar(0, 1, f"x_{u}_{v}")

    # Hàm mục tiêu: tổng chi phí các cạnh được chọn
    solver.Minimize(solver.Sum(cost * x[(u, v)] for (u, v, cost) in edges))
    
    # Ràng buộc: tại đỉnh s, chỉ có 1 cạnh xuất phát được chọn
    solver.Add(solver.Sum(x[(s, v)] for (u, v, _) in edges if u == s) == 1)
    # Ràng buộc: tại đỉnh t, chỉ có 1 cạnh đến được chọn
    solver.Add(solver.Sum(x[(u, t)] for (u, v, _) in edges if v == t) == 1)
    
    # Với các đỉnh còn lại (không phải s, t): nếu đỉnh đó được thăm thì số cạnh vào bằng số cạnh ra và không vượt quá 1
    for v in range(n):
        if v == s or v == t:
            continue
        in_edges = [x[(u, v)] for (u, vv, _) in edges if vv == v]
        out_edges = [x[(v, w)] for (v0, w, _) in edges if v0 == v]
        # Nếu đỉnh được thăm thì tổng vào = tổng ra = 1; nếu không thăm thì = 0.
        solver.Add(solver.Sum(in_edges) == solver.Sum(out_edges))
        solver.Add(solver.Sum(in_edges) <= 1)

    # Ràng buộc loại bỏ chu trình phụ (sử dụng ràng buộc MTZ)
    # Biến u[v] dùng để đánh số thứ tự thăm các đỉnh trên đường đi
    u_vars = {}
    for v in range(n):
        if v == s:
            u_vars[v] = solver.NumVar(0, 0, f"u_{v}")  # Đặt u[s] = 0
        else:
            u_vars[v] = solver.NumVar(0, n - 1, f"u_{v}")
    
    # Với mỗi cạnh (i, j) với i, j khác s, thêm ràng buộc:
    # u[i] - u[j] + (n-1)*x[i,j] <= n-2
    for (i, j, _) in edges:
        if i == s or j == s:
            continue
        solver.Add(u_vars[i] - u_vars[j] + (n - 1) * x[(i, j)] <= n - 2)
    
    # Ràng buộc các cặp cạnh cấm: không được chọn đồng thời cả 2 cạnh
    for (edge1, edge2) in forbidden_pairs:
        # Chỉ thêm ràng buộc nếu cả 2 cạnh có tồn tại trong tập E
        if edge1 in x and edge2 in x:
            solver.Add(x[edge1] + x[edge2] <= 1)
    
    # Giải mô hình
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        total_cost = solver.Objective().Value()
        # In kết quả: tổng chi phí của đường đi
        print(int(total_cost))
    else:
        # Không tìm thấy lời giải khả thi
        print(-1)

if __name__ == '__main__':
    main()
