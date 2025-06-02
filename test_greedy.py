import os
import time
import subprocess
import platform


# Go compile yourself guys
def get_executable_name():
    system = platform.system().lower()
    if system == "windows":
        return "tsp_time_window_greedy.exe"
    return "./tsp_time_window_greedy"


def run_test(test_dir):
    input_file = os.path.join(test_dir, "input.in")
    expected_output = os.path.join(test_dir, "output.out")
    result_output = os.path.join(test_dir, "sol_greedy.out")

    with open(expected_output, "r") as f:
        expected = f.read().strip()

    with open(input_file, "r") as f:
        input_data = f.read()

    start_time = time.time()
    process = subprocess.Popen(
        [get_executable_name()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    output, _ = process.communicate(input=input_data)
    solve_time = time.time() - start_time

    output_lines = output.strip().split("\n")
    cost = "0"
    violations = "0"
    actual = "NOT_FEASIBLE"
    if output_lines:
        for line in output_lines:
            if line.startswith("Cost:"):
                cost = line.split("Cost:")[1].strip()
            elif line.startswith("Violations:"):
                violations = line.split("Violations:")[1].strip()
            elif (
                line.strip()
                and not line.startswith("Cost:")
                and not line.startswith("Violations:")
            ):
                actual = line.strip()

    results = []
    results.append(f"Status: {'FEASIBLE' if actual == expected else 'INFEASIBLE'}")
    results.append(f"Best sol: [{', '.join(map(str, actual.split()))}]")
    results.append(f"Violations: {violations}")
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
