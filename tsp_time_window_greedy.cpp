#include <algorithm>
#include <iostream>
#include <vector>

using std::pair;
using std::vector;

int main()
{
    int n;
    std::cin >> n;

    // time_windows[i] = {e_i, l_i} - cửa sổ thời gian của khách hàng i
    vector<pair<int, int>> time_windows(n);
    // service_times[i] = d_i - thời gian phục vụ khách hàng i
    vector<int> service_times(n);

    // Đọc thông tin khách hàng
    for (int i = 0; i < n; ++i) {
        int e, l, d;
        std::cin >> e >> l >> d;
        time_windows[i] = { e, l };
        service_times[i] = d;
    }

    // Đọc ma trận thời gian di chuyển
    vector<vector<int>> travel_times(n + 1, vector<int>(n + 1));
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            std::cin >> travel_times[i][j];
        }
    }

    // Tạo danh sách khách hàng với chỉ số
    vector<int> customers;
    for (int i = 0; i < n; ++i) {
        customers.push_back(i + 1);
    }

    // Đây là thuật toán greedy: ưu tiên khách hàng có thời gian bắt đầu sớm nhất
    // https://web.archive.org/web/20231004211702/https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.5196&rep=rep1&type=pdf
    std::sort(customers.begin(), customers.end(), [&](int a, int b) {
        return time_windows[a - 1].first < time_windows[b - 1].first;
    });

    // Tính tổng chi phí di chuyển
    int total_travel_time = 0;
    int current_time = 0;
    int current_location = 0;
    int violations = 0;

    for (int customer : customers) {
        // Thời gian di chuyển
        int travel_time = travel_times[current_location][customer];
        total_travel_time += travel_time;

        // Thời gian đến khách hàng
        current_time += travel_time;

        // Nếu đến sớm, đợi đến thời gian bắt đầu
        if (current_time < time_windows[customer - 1].first) {
            current_time = time_windows[customer - 1].first;
        }

        // Kiểm tra vi phạm: bắt đầu phục vụ sau latest time
        if (current_time > time_windows[customer - 1].second) {
            violations++;
        }

        // Thời gian phục vụ
        current_time += service_times[customer - 1];

        // Cập nhật vị trí hiện tại
        current_location = customer;
    }

    // Chi phí quay về kho
    total_travel_time += travel_times[current_location][0];

    std::cout << "Cost: " << total_travel_time << "\n";
    std::cout << "Violations: " << violations << "\n";
    std::cout << n << "\n";
    for (int i = 0; i < n; ++i) {
        std::cout << customers[i];
        if (i < n - 1)
            std::cout << " ";
    }
    std::cout << "\n";

    return 0;
}