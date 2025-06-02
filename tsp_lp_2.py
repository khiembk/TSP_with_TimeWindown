from ortools.linear_solver import pywraplp


def read_input():
    n = int(input())
    time_windows = []
    service_times = [0]

    for _ in range(n):
        e, l, d = map(int, input().split())
        time_windows.append((e, l))
        service_times.append(d)

    travel_times = []
    for _ in range(n + 1):
        travel_times.append(list(map(int, input().split())))

    return n, time_windows, service_times, travel_times


def solve_tsp_milp(n, time_windows, service_times, travel_times):
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    max_time = max(l for e, l in time_windows)
    big_m = max_time + sum(service_times) + max(max(row) for row in travel_times)

    # Biến quyết định
    # x[i][j] = 1 nếu đi từ i đến j
    x = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:  # Loại bỏ self-loop
                x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    # Biến thời gian bắt đầu phục vụ
    t = [solver.NumVar(0, big_m, f"t_{i}") for i in range(n + 1)]

    # Ràng buộc luồng
    # Mỗi điểm có đúng 1 cạnh vào và 1 cạnh ra
    for i in range(n + 1):
        # Cạnh vào
        solver.Add(sum(x[j, i] for j in range(n + 1) if j != i) == 1)
        # Cạnh ra
        solver.Add(sum(x[i, j] for j in range(n + 1) if i != j) == 1)

    # Xuất phát từ depot tại thời điểm 0
    solver.Add(t[0] == 0)

    # Ràng buộc thời gian
    for i in range(n + 1):
        for j in range(1, n + 1):  # Không tính depot là đích
            if i != j:
                solver.Add(
                    t[j]
                    >= t[i]
                    + service_times[i]
                    + travel_times[i][j]
                    - big_m * (1 - x[i, j])
                )

    # Ràng buộc time windows
    for i in range(1, n + 1):
        solver.Add(t[i] >= time_windows[i - 1][0])
        solver.Add(t[i] <= time_windows[i - 1][1])

    # Loại bỏ Subtour(MTZ)
    u = [solver.IntVar(0, n, f"u_{i}") for i in range(n + 1)]
    solver.Add(u[0] == 0)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                solver.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

    # Hàm mục tiêu:
    objective = solver.Sum(
        x[i, j] * travel_times[i][j]
        for i in range(n + 1)
        for j in range(n + 1)
        if i != j
    )
    solver.Minimize(objective)

    solver.SetTimeLimit(120000)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print(f"Cost: {int(solver.Objective().Value())}")
        # Trích xuất lộ trình
        route = []
        current = 0
        visited = {0}

        while len(route) < n:
            for j in range(n + 1):
                if j not in visited and x[current, j].solution_value() > 0.5:
                    if j != 0:
                        route.append(j)
                    visited.add(j)
                    current = j
                    break

        print(n)
        print(" ".join(map(str, route)))
    else:
        print("NOT_FEASIBLE")


if __name__ == "__main__":
    n, time_windows, service_times, travel_times = read_input()
    solve_tsp_milp(n, time_windows, service_times, travel_times)
