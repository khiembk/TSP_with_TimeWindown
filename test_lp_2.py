import os
import sys
import time
from io import StringIO
from tsp_lp_2 import read_input, solve_tsp_milp


def run_test(test_dir):
    input_file = os.path.join(test_dir, "input.in")
    expected_output = os.path.join(test_dir, "output.out")
    result_output = os.path.join(test_dir, "sol_lp_2.out")

    with open(expected_output, "r") as f:
        expected = f.read().strip()

    with open(input_file, "r") as f:
        sys.stdin = StringIO(f.read())
        sys.stdout = StringIO()
        n, time_windows, service_times, travel_times = read_input()
        start_time = time.time()
        solve_tsp_milp(n, time_windows, service_times, travel_times)
        solve_time = time.time() - start_time
        output_lines = sys.stdout.getvalue().strip().split("\n")

        cost = "0"
        actual = "NOT_FEASIBLE"
        if output_lines:
            for line in output_lines:
                if line.startswith("Cost:"):
                    cost = line.split("Cost:")[1].strip()
                elif line.strip() and not line.startswith("Cost:"):
                    actual = line.strip()

    results = []
    results.append(f"Status: {'FEASIBLE' if actual == expected else 'INFEASIBLE'}")
    results.append(f"Best sol: [{', '.join(map(str, actual.split()))}]")
    results.append(f"Cost: {cost}")
    results.append(f"Solve Time: {solve_time:.2f} secs")

    with open(result_output, "w") as f:
        for result in results:
            f.write(result + "\n")

    return actual == expected


def main():
    for i in range(1, 12):
        test_dir = f"tests/test{i}"
        run_test(test_dir)


if __name__ == "__main__":
    main()
