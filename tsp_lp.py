from ortools.linear_solver import pywraplp


def read_input():
    n = int(input())
    time_arrivals = []
    times = []
    time_deleve = []
    max_time = 0
    time_deleve.append(0)

    for _ in range(n):
        line = list(map(int, input().split()))
        time_arrivals.append(line)
        time_deleve.append(line[2])
        max_time = max(max_time, line[1])

    for _ in range(n + 1):
        line = list(map(int, input().split()))
        times.append(line)

    return n, time_arrivals, times, max_time, time_deleve


def solve_tsp_lp(n, time_arrivals, times, max_time, time_deleve):
    solver = pywraplp.Solver.CreateSolver("SCIP")

    x = [[solver.BoolVar(f"x_{i}_{j}") for j in range(n + 1)] for i in range(n + 1)]

    for i in range(n + 1):
        solver.Add(sum(x[j][i] for j in range(n + 1)) == 1)
        solver.Add(sum(x[i][j] for j in range(n + 1)) == 1)

    t = [solver.IntVar(0, max_time, f"t_{i}") for i in range(n + 1)]

    solver.Add(t[0] == 0)

    for i in range(n + 1):
        for j in range(1, n + 1):
            solver.Add(
                t[j] >= t[i] + time_deleve[i] + times[i][j] - max_time * (1 - x[i][j])
            )

    for i in range(1, n + 1):
        solver.Add(t[i] >= time_arrivals[i - 1][0])
        solver.Add(t[i] <= time_arrivals[i - 1][1])

    Totalcost = sum(x[i][j] * times[i][j] for j in range(n) for i in range(n))

    solver.Minimize(Totalcost)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Cost: ", int(Totalcost.solution_value()))
        path = []
        current = 0
        visited = set()
        while current not in visited:
            path.append(current)
            visited.add(current)
            for j in range(n + 1):
                if x[current][j].solution_value() > 0.5:
                    current = j
                    break

        city_tour = path[1:]
        # Output as required
        print(n)
        print(" ".join(map(str, city_tour)))
    else:
        print("NOT_FEASIBLE")


if __name__ == "__main__":
    n, time_arrivals, times, max_time, time_deleve = read_input()
    # print(time_arrivals)
    solve_tsp_lp(n, time_arrivals, times, max_time, time_deleve)
