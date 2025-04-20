# Hướng dẫn sử dụng
## Mô hình hóa
1. Mỗi đối tượng trên đường đi sẽ được miêu tả bằng lớp `Client`.
2. Một bài toán TSP Time Window sẽ được mô hình hóa bằng lớp `TSPTWProblem`.
3. Mỗi ràng buộc có thể được thêm vào problem để tính số vi phạm (violations) đều phải implement `Constraint`.
4. Một lời giải hoán vị được mô hình hóa bằng lớp `PermuSolution`.
5. Một thuật toán heuristic, meta-heuristic sẽ implement `Solver` và cài đặt phương thức `solve` (với tham số tùy chọn).
## Các thuật toán đã cài đặt
### Các thuật toán Local Search
Được cài đặt trong [ls.py](./ls.py), gồm 2 thuật toán
- Hill Climbing Search
- Simulated Annealing
Cả 2 thuật toán đều có module restart khi gặp tối ưu cục bộ.
### Thuật toán GA
Được cài đặt trong [ga.py](./ga.py). Implement kiến trúc GA thông thường.
### Thuật toán ALNS
Thuật toán ALNS được cài đặt với các toán tử
- Remove: Random, Worst, Shaw
- Insert: Random, Greedy, Regret
Có adaptive, tự điều chỉnh trọng số các toán tử.

Ngoài ra cung cấp các cluster-base operator:
- Remove: Time Window Cluster
- Insert: Time Window Cluster
## Triển khai
Chạy thử trên notebook (có thể vào đó xem cách sử dụng)
[Test TSP TW](https://www.kaggle.com/code/trietp1253201581/test-tsp-tw)