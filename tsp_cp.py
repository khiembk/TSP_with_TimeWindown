from ortools.sat.python import cp_model


def read_input():
    n = int(input())
    time_windows = []
    service_times = []

    for _ in range(n):
        e, l, d = map(int, input().split())
        time_windows.append((e, l))
        service_times.append(d)

    travel_times = []
    for _ in range(n + 1):
        row = list(map(int, input().split()))
        travel_times.append(row)

    return n, time_windows, service_times, travel_times


def solve_tsp_cp(n, time_windows, service_times, travel_times):
    model = cp_model.CpModel()

    # Tính thời gian tối đa cho miền giá trị
    max_time = (
        max(l for _, l in time_windows)
        + sum(service_times)
        + sum(max(row) for row in travel_times)
    )

    # Biến quyết định: x[i][j] = 1 nếu đi từ điểm i đến điểm j
    x = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    # Biến thời gian đến và thời gian bắt đầu phục vụ
    arrival_time = [
        model.NewIntVar(0, max_time, f"arrival_time_{i}") for i in range(n + 1)
    ]
    start_time = [model.NewIntVar(0, max_time, f"start_time_{i}") for i in range(n + 1)]

    # Ràng buộc luồng: mỗi điểm được thăm đúng một lần
    for j in range(1, n + 1):
        model.Add(sum(x[i, j] for i in range(n + 1) if i != j) == 1)

    for i in range(1, n + 1):
        model.Add(sum(x[i, j] for j in range(n + 1) if i != j) == 1)

    # Kho có đúng một cạnh ra và một cạnh vào
    model.Add(sum(x[0, j] for j in range(1, n + 1)) == 1)
    model.Add(sum(x[j, 0] for j in range(1, n + 1)) == 1)

    # Xuất phát từ kho tại thời điểm 0
    model.Add(arrival_time[0] == 0)
    model.Add(start_time[0] == 0)

    # Ràng buộc thời gian di chuyển giữa các điểm
    for i in range(n + 1):
        for j in range(1, n + 1):
            if i != j:
                service_time_i = service_times[i - 1] if i > 0 else 0
                model.Add(
                    arrival_time[j]
                    >= start_time[i]
                    + service_time_i
                    + travel_times[i][j]
                    - max_time * (1 - x[i, j])
                )

    # Thời gian bắt đầu phục vụ phải thỏa mãn time window
    for i in range(1, n + 1):
        model.AddMaxEquality(start_time[i], [arrival_time[i], time_windows[i - 1][0]])
        model.Add(start_time[i] <= time_windows[i - 1][1])

    # Loại bỏ chu trình con (MTZ)
    u = [model.NewIntVar(0, n, f"u_{i}") for i in range(n + 1)]
    model.Add(u[0] == 0)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                model.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

    # Hàm mục tiêu: tối thiểu hóa tổng thời gian di chuyển
    total_travel_time = sum(
        x[i, j] * travel_times[i][j]
        for i in range(n + 1)
        for j in range(n + 1)
        if i != j
    )

    model.Minimize(total_travel_time)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Trích xuất lộ trình từ nghiệm
        route = []
        current = 0
        visited = set([0])

        while len(route) < n:
            for j in range(n + 1):
                if j not in visited and solver.Value(x[current, j]) == 1:
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
    solve_tsp_cp(n, time_windows, service_times, travel_times)
