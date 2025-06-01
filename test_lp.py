import os
import sys
import time
from io import StringIO
from tsp_lp import read_input, solve_tsp_lp


def run_test(test_dir):
    input_file = os.path.join(test_dir, "input.in")
    expected_output = os.path.join(test_dir, "output.out")
    result_output = os.path.join(test_dir, "sol_lp.out")

    with open(expected_output, "r") as f:
        expected = f.read().strip()

    with open(input_file, "r") as f:
        sys.stdin = StringIO(f.read())
        sys.stdout = StringIO()
        n, time_arrivals, times, max_time, time_deleve = read_input()
        start_time = time.time()
        solve_tsp_lp(n, time_arrivals, times, max_time, time_deleve)
        solve_time = time.time() - start_time
        output_lines = sys.stdout.getvalue().strip().split("\n")
        actual = output_lines[-1] if output_lines else "NOT_FEASIBLE"

    results = []
    results.append(f"Status: {'FEASIBLE' if actual == expected else 'INFEASIBLE'}")
    results.append(f"Best sol: [{', '.join(map(str, actual.split()))}]")
    results.append(f"Solve Time: {solve_time:.2f} secs")

    with open(result_output, "w") as f:
        for result in results:
            f.write(result + "\n")

    return actual == expected


def main():
    for i in range(1, 4):
        test_dir = f"tests/test{i}"
        run_test(test_dir)


if __name__ == "__main__":
    main()
