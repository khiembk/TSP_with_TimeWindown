import sys
import random
import math

# Đọc input
N = int(input())
e = [0] * (N + 1)  #
l = [0] * (N + 1)
d = [0] * (N + 1)
t = [[0] * (N + 1) for _ in range(N + 1)]

e[0] = 0
l[0] = float('inf')
d[0] = 0

for i in range(1, N + 1):
    e[i], l[i], d[i] = map(int, input().split())

for i in range(N + 1):
    t[i] = list(map(int, input().split()))


def is_feasible(route):
    time = 0
    for i in range(len(route) - 1):
        curr = route[i]
        next_node = route[i + 1]
        time = max(e[next_node], time + d[curr] + t[curr][next_node])
        if time > l[next_node]:
            return False
    return True


def calculate_cost(route):
    cost = 0
    for i in range(len(route) - 1):
        cost += t[route[i]][route[i + 1]]
    return cost


def initialize_solution():
    route = [0, 0]
    unvisited = list(range(1, N + 1))

    unvisited.sort(key=lambda x: e[x])

    while unvisited:
        best_cost = float('inf')
        best_route = None
        customer_to_remove = None

        for customer in unvisited:
            for i in range(1, len(route)):
                new_route = route[:i] + [customer] + route[i:]
                if is_feasible(new_route):
                    cost = calculate_cost(new_route)
                    if cost < best_cost:
                        best_cost = cost
                        best_route = new_route
                        customer_to_remove = customer

        if best_route is None:
            return greedy_initialize()

        route = best_route
        unvisited.remove(customer_to_remove)

    return route


def greedy_initialize():
    route = [0]
    unvisited = list(range(1, N + 1))

    while unvisited:
        best_cost = float('inf')
        best_customer = None
        for customer in unvisited:
            new_route = route + [customer]
            if is_feasible(new_route + [0]):  # Kiểm tra tạm thời
                cost = t[route[-1]][customer]
                if cost < best_cost:
                    best_cost = cost
                    best_customer = customer
        if best_customer is None:
            return None
        route.append(best_customer)
        unvisited.remove(best_customer)

    route.append(0)
    if is_feasible(route):
        return route
    return None


def local_search():
    solution = initialize_solution()
    if solution is None:
        return None

    best_solution = solution
    best_cost = calculate_cost(solution)

    T = 1000.0
    T_min = 0.1
    alpha = 0.995
    max_iterations = 1000

    while T > T_min and max_iterations > 0:
        # Tạo lân cận
        for _ in range(N):
            op = random.choice(['swap', '2-opt', 'relocate'])
            neighbor = solution.copy()

            if op == 'swap':
                i, j = random.sample(range(1, N), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            elif op == '2-opt':
                i, j = sorted(random.sample(range(1, N), 2))
                neighbor[i:j + 1] = reversed(neighbor[i:j + 1])

            elif op == 'relocate':
                i = random.randint(1, N)
                j = random.randint(1, N)
                if i != j:
                    customer = neighbor.pop(i)
                    neighbor.insert(j, customer)

            if is_feasible(neighbor):
                cost = calculate_cost(neighbor)
                delta = cost - best_cost

                if delta <= 0 or random.random() < math.exp(-delta / T):
                    solution = neighbor
                    if cost < best_cost:
                        best_solution = solution
                        best_cost = cost

        T *= alpha
        max_iterations -= 1

    return best_solution


solution = local_search()

if solution is None:
    print("No feasible solution found")
else:
    print(N)
    print(" ".join(map(str, solution[1:-1])))