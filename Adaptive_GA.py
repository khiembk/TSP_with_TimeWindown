import random
from copy import deepcopy

def read_input():
    try:
        n = int(input())
        if not 1 <= n <= 1000:
            raise ValueError("N must be between 1 and 1000")
        
        time_arrivals = []
        times = []
        max_time = 0
        
        for _ in range(n):
            e, l, d = map(int, input().split())
            if e < 0 or l < e or d < 0:
                raise ValueError("Invalid time window or service time")
            time_arrivals.append([e, l, d])
            max_time = max(max_time, l)
        
        for _ in range(n+1):
            row = list(map(int, input().split()))
            if len(row) != n+1 or any(t < 0 for t in row):
                raise ValueError("Invalid travel time matrix row")
            times.append(row)
        
        return n, time_arrivals, times, max_time
    except ValueError as e:
        raise ValueError(f"Input error: {str(e)}")

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
        chromosome = generate_random_chromosome(n)
        population.append(chromosome)
    return population

def generate_random_chromosome(n):
    chromosome = list(range(1, n+1))
    random.shuffle(chromosome)
    if len(chromosome) != n or len(set(chromosome)) != n or not all(1 <= x <= n for x in chromosome):
        raise ValueError(f"Invalid chromosome generated: {chromosome}")
    return chromosome

def tournament_selection(population, fitnesses, tournament_size=3):
    if not population or len(population) < tournament_size:
        raise ValueError("Population too small for tournament selection")
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    return min(selected, key=lambda x: x[1])[0]

def order_crossover(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={parent1}, parent2={parent2}")
    
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[start:end+1] = parent1[start:end+1]
    
    # Filter remaining genes from parent2
    remaining = [x for x in parent2 if x not in child[start:end+1]]
    if len(remaining) != n - (end - start + 1):
        raise ValueError(f"Invalid remaining genes: {remaining}, expected length {n - (end - start + 1)}")
    
    idx = 0
    for i in range(n):
        if child[i] == -1:
            if idx >= len(remaining):
                raise ValueError(f"Index out of range: idx={idx}, remaining={remaining}")
            child[i] = remaining[idx]
            idx += 1
    
    if len(set(child)) != n or any(x < 1 or x > n for x in child):
        raise ValueError(f"Invalid child generated: {child}")
    return child

def pmx_crossover(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={parent1}, parent2={parent2}")
    
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[start:end+1] = parent1[start:end+1]
    mapping = {parent1[i]: parent2[i] for i in range(start, end+1)}
    reverse_mapping = {parent2[i]: parent1[i] for i in range(start, end+1)}
    
    for i in range(n):
        if i < start or i > end:
            gene = parent2[i]
            while gene in child:
                gene = mapping.get(gene, gene)
                if gene in child:
                    gene = reverse_mapping.get(gene, gene)
            child[i] = gene
    
    if len(set(child)) != n or any(x < 1 or x > n for x in child):
        raise ValueError(f"Invalid child generated: {child}")
    return child

def cycle_crossover(parent1, parent2):
    n = len(parent1)
    if len(parent2) != n or len(set(parent1)) != n or len(set(parent2)) != n:
        raise ValueError(f"Invalid parents: parent1={parent1}, parent2={parent2}")
    
    child = [-1] * n
    visited = [False] * n
    pos = 0
    cycle = []
    while not visited[pos]:
        cycle.append(pos)
        visited[pos] = True
        gene = parent1[pos]
        pos = parent2.index(gene)
    
    for i in cycle:
        child[i] = parent1[i]
    
    for i in range(n):
        if child[i] == -1:
            child[i] = parent2[i]
    
    if len(set(child)) != n or any(x < 1 or x > n for x in child):
        raise ValueError(f"Invalid child generated: {child}")
    return child

def mutate(chromosome, n, time_arrivals, times, cur_ep, total_ep, delta=20, mutation_rate=0.01, max_time=10):
    if random.random() >= mutation_rate:
        return chromosome
    
    old_penalty = compute_penalty(n, time_arrivals, times, chromosome)
    original_chromosome = chromosome.copy()
    attempts = 0
    
    while attempts < max_time:
        new_chromosome = chromosome.copy()
        i, j = random.sample(range(len(new_chromosome)), 2)
        new_chromosome[i], new_chromosome[j] = new_chromosome[j], new_chromosome[i]
        
        new_penalty = compute_penalty(n, time_arrivals, times, new_chromosome)
        if new_penalty <= old_penalty + delta:
            return new_chromosome
        attempts += 1
    
    return original_chromosome

def select_population(n, time_arrivals, times, cur_ep, total_ep, population, new_population, pop_size, reserve_pop_size):
    combined = [(deepcopy(chrom), compute_fitness(n, time_arrivals, times, chrom, cur_ep, total_ep)) 
                for chrom in population + new_population]
    
    combined.sort(key=lambda x: x[1])
    new_population = [chrom for chrom, _ in combined[:pop_size]]
    reserve_pop = combined[pop_size:pop_size + reserve_pop_size]
    
    return new_population, reserve_pop

def genetic_algorithm(n, time_arrivals, times, max_time, pop_size=150, reserve_pop_size=30, total_ep=800, mutation_rate=0.01, p_reserve=0.1, random_chrom_prob=0.1):
    population = generate_population(n, pop_size)
    reserve_pop = [(chrom, compute_fitness(n, time_arrivals, times, chrom, 0, total_ep)) 
                   for chrom in generate_population(n, reserve_pop_size)]
    crossover_methods = order_crossover
    best_solution = None
    best_fitness = float('inf')
    
    for ep in range(total_ep):
        fitnesses = [compute_fitness(n, time_arrivals, times, chrom, ep, total_ep) for chrom in population]
        
        min_fitness = min(fitnesses)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[fitnesses.index(min_fitness)].copy()
        
        new_population = []
        for _ in range(pop_size):
            try:
                
                parent1 = tournament_selection(population, fitnesses)
                
                if random.random() < p_reserve*(ep/total_ep) and reserve_pop:
                    parent2, _ = random.choice(reserve_pop)
                else:
                    parent2 = tournament_selection(population, fitnesses)
                
                #crossover = random.choice(crossover_methods)
                crossover = crossover_methods
                child = crossover(parent1, parent2)
                child = mutate(child, n, time_arrivals, times, ep, total_ep, mutation_rate=mutation_rate)
                new_population.append(child)
            except Exception as e:
                print(f"Error in crossover/mutation: {str(e)}")
                continue
        
        population, reserve_pop = select_population(n, time_arrivals, times, ep, total_ep, population, new_population, pop_size, reserve_pop_size)
    
    return best_solution

if __name__ == "__main__":
    try:
        n, time_arrivals, times, max_time = read_input()
        best_route = genetic_algorithm(n, time_arrivals, times, max_time)
        print(n)
        cost = compute_cost(n, time_arrivals, times, best_route)
        print("solution value", cost if cost != float('inf') else "Invalid solution")
        print(*best_route)
    except ValueError as e:
        print(f"Error: {str(e)}")