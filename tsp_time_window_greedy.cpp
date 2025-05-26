#include <algorithm>
#include <iostream>
#include <vector>

using std::vector;

struct Customer {
    int index, startTime, endTime, duration;
};

int main()
{
    int N;
    std::cin >> N;

    vector<Customer> customers(N);

    // Đọc thông tin khách hàng
    for (int i = 0; i < N; ++i) {
	    std::cin >> customers[i].startTime >> customers[i].endTime >> customers[i].duration;
        customers[i].index = i + 1;
    }

    // Đọc ma trận thời gian di chuyển
    vector<vector<int>> travelTimes(N + 1, vector<int>(N + 1));
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
		std::cin >> travelTimes[i][j];
        }
    }

    // Sắp xếp các khách hàng dựa trên thời gian bắt đầu giao hàng
    std::sort(customers.begin(), customers.end(), [](const Customer& a, const Customer& b) {
        return a.startTime < b.startTime;
    });

    // Lập lộ trình giao hàng
    vector<int> deliveryRoute;
    for (const auto& customer : customers) {
        deliveryRoute.push_back(customer.index);
    }

    // In kết quả
    std::cout << N << "\n";
    for (const auto& point : deliveryRoute) {
	    std::cout << point << " ";
    }

    return 0;
}
