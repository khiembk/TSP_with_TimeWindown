import random

def read_input():
    n = int(input())
    
    
    time_arrivals = []
    times = []
    max_time = 0
    
    # Read time windows and service times
    for _ in range(n):
        e, l, d = map(int, input().split())
        time_arrivals.append([e, l, d])
        max_time = max(max_time, l)
    
    # Read travel time matrix
    for _ in range(n+1):
        row = list(map(int, input().split()))
        if len(row) != n+1:
            raise ValueError("Invalid travel time matrix row")
        times.append(row)
    
    return n, time_arrivals, times, max_time

def compute_cost(n, time_arrivals, times, chromosome):
    if len(chromosome) != n or not all(1 <= x <= n for x in chromosome) or len(set(chromosome)) != n:
        return float('inf')
    
    cost = 0
    current = 0
    for i in range(n):
        cost += times[current][chromosome[i]]
        current = chromosome[i]
    
    cost += times[current][0]
    return cost

def compute_penalty(n, time_arrivals, times, chromosome):
    if len(chromosome) != n or not all(1 <= x <= n for x in chromosome) or len(set(chromosome)) != n:
        return float('inf')
    
    time_arrive = []
    cur_time = 0
    cur_node = 0
    for i in range(n):
        customer = chromosome[i]
        cur_time += times[cur_node][customer]
        if cur_time < time_arrivals[customer-1][0]:
            cur_time = time_arrivals[customer-1][0]
        time_arrive.append(cur_time)
        cur_time += time_arrivals[customer-1][2]
        cur_node = customer
    
    penalty = 0
    for i in range(n):
        customer = chromosome[i]
        if time_arrive[i] > time_arrivals[customer-1][1]:
            penalty += time_arrive[i] - time_arrivals[customer-1][1]
    
    return penalty

def compute_fitness(n, time_arrivals, times, chromosome, cur_ep, total_ep, epsilon=0.01, big_M=100000):
    cost = compute_cost(n, time_arrivals, times, chromosome)
    penalty = compute_penalty(n, time_arrivals, times, chromosome)
    return cost + big_M * (cur_ep / total_ep + epsilon) * penalty

def generate_population(n, pop_size):
    population = []
    for _ in range(pop_size):
        chromosome = list(range(1, n+1))
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    return min(selected, key=lambda x: x[1])[0]

def order_crossover(parent1, parent2):
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[start:end+1] = parent1[start:end+1]
    remaining = [x for x in parent2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = remaining[idx]
            idx += 1
    return child

def mutate(chromosome, mutation_rate=0.01):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def genetic_algorithm(n, time_arrivals, times, max_time, pop_size=100, total_ep=1000, mutation_rate=0.01):
    population = generate_population(n, pop_size)
    best_solution = None
    best_fitness = float('inf')
    
    for ep in range(total_ep):
        fitnesses = [compute_fitness(n, time_arrivals, times, chrom, ep, total_ep) for chrom in population]
        
        # Track best solution
        min_fitness = min(fitnesses)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[fitnesses.index(min_fitness)].copy()
        
        # Create new population
        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = order_crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    return best_solution

if __name__ == "__main__":
    n, time_arrivals, times, max_time = read_input()
    best_route = genetic_algorithm(n, time_arrivals, times, max_time)
    print(n)
    print(*best_route)