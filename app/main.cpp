
#include <omp.h>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <execution>
#include <vector>
#include <numeric>
#include <random>
#include <unistd.h>
#include <variant>
#include <ranges>
#include <queue>
#include <tbb/global_control.h>
#include "oblivious_operations.hpp"
#include "iterator_stride.hpp"
#include "ocompact.hpp"
#include "ohash_tiers.hpp"
#include "omap.hpp"
#include "oshuffle.hpp"
#include "types.hpp"
#include "timer.hpp"
#include "hash_planner.hpp"
#include "oram.hpp"
#include "graphs.hpp"
#include <iomanip>
#include <unordered_set>
#include <set>

int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    int test_cases = 1;
    // uint32_t n = MAJOR_BIN_SIZE * EPSILON_INV * 4;
    uint32_t n = argc >= 2 ? std::atoi(argv[1]) : 1 << 21; // 1 << 16;
    std::cout << "n: " << n << std::endl;
    constexpr int B = 12;
    while (test_cases--)
    {
        Timer t;
        // std::vector<int> data(n);
        // for (uint32_t i = 0; i < n; i++)
        //     data[i] = i;
        std::vector<ORAM::Block<uint32_t, B>> data(n);
        for (uint32_t i = 0; i < n; i++)
            data[i].id = i;
        std::vector<uint8_t> flags(n);
        std::shuffle(data.begin(), data.end(), gen);
        std::cout << "Preprocessing: " << t.get_interval_time() << " s" << std::endl;
        ORAM::ObliviousRAM<uint32_t, ORAM::Block<uint32_t, B>> oram(data.begin(), data.end());
        auto max_mem_usage = getMemoryUsage();
        std::cout << "total: " << n * sizeof(oram[0]) / 1024 << " KB, memory usage: " << getMemoryUsage() << " KB" << ", ratio: " << 1.0 * getMemoryUsage() / n / sizeof(data[0]) * 1024 << std::endl;
        std::cout << "Init: " << t.get_interval_time() << " s" << std::endl;
        // ProfilerStart("test_capture.prof");
        uint32_t idx;
        std::uniform_int_distribution<uint32_t> dis(0, n - 1);
        constexpr uint32_t T = 1;
        for (int _ = 0; _ < T; _++)
            for (uint32_t i = 0; i < n; i++)
            {
                // idx = dis(gen);
                idx = i;
                assert(oram[idx] == data[idx]);
                // max_mem_usage = std::max(max_mem_usage, getMemoryUsage());
            }
        std::cout << "total: " << n * sizeof(oram[0]) / 1024 << " KB, memory usage: " << getMemoryUsage() << " KB" << ", ratio: " << 1.0 * getMemoryUsage() / n / sizeof(data[0]) * 1024 << std::endl;
        // ProfilerStop();
        auto _ = t.get_interval_time();
        std::cout << "ObliviousRAM: " << _ << " s" << ", per op: " << _ / T / n * 1000000 << " us" << std::endl;
    }
}
