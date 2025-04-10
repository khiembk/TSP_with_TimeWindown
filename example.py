from common import read_console, TimeWindowConstraint, MustFollowConstraint, PermuSolution, Solver, read_input_file
import alns
from common import HNNInitOperator, HybridInitOperator

problem = read_input_file('tests/test6/input.in')

problem.add_constraint(TimeWindowConstraint())
# problem.add_constraint(MustFollowConstraint())

init_hnn = HybridInitOperator(prob=1, problem=problem, hn_ratio=0.12)

remove_random = alns.RandomRemoveOperator(prob=0.3, problem=problem)
remove_worst = alns.WorstRemoveOperator(prob=0.1, problem=problem)
remove_shaw = alns.ShawRemoveOperator(prob=0.6, problem=problem)

insert_greedy = alns.GreedyInsertOperator(prob=0.3, problem=problem)
insert_regret = alns.RegretInsertOperator(prob=0.7, problem=problem)

solver = alns.ALNSSolver(problem, lr=0.1)



solver.add_init_opr(init_hnn)
solver.add_remove_opr(remove_random)
solver.add_remove_opr(remove_worst)
solver.add_remove_opr(remove_shaw)
solver.add_insert_opr(insert_greedy)
solver.add_insert_opr(insert_regret)
import random
random.seed(42)
status = solver.solve(debug=True, num_iters=30, remove_fraction=0.2,
                      insert_idx_selected=30, update_weight_freq=0.2)

print(status)

print(solver.best_solution)
print(solver.best_violations)
print(solver.best_penalty)
print(solver.best_cost)
    


