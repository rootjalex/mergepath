#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

#include "partition.h"

// Generates a sorted vector of size n with random values in [min_val, max_val]
template<typename T>
std::vector<T> generate_sorted_vector(size_t n, T min_val = 0, T max_val = 1000) {
    static std::mt19937 rng(42); // for reproducibility
    std::uniform_real_distribution<T> dist(min_val, max_val);

    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    std::sort(v.begin(), v.end());
    return v;
}

// Sweep sizes for 2D test cases
template<typename value_t, typename index_t>
void test_partition2D_sweep(const index_t n) {
    for (index_t nA_exp = 0; nA_exp < n; nA_exp++) {
        const index_t countA = index_t(1ull << nA_exp);
        for (index_t nB_exp = 0; nB_exp < n; nB_exp++) {
            const index_t countB = index_t(1ull << nB_exp);
            auto A = generate_sorted_vector<value_t>(countA);
            auto B = generate_sorted_vector<value_t>(countB);
            index_t max_count = countA + countB;
            for (index_t count = 0; count < max_count; ++count) {
                auto [l_a, l_b] = partition2D<value_t, index_t>(A.data(), countA, B.data(), countB, count);

                if (l_a < 0 || l_a > countA || l_b < 0 || l_b > countB) {
                    std::cerr << "out of range results: nA=" << countA
                              << " nB=" << countB << " count=" << count
                              << " -> l_a=" << l_a << " l_b=" << l_b << "\n";
                    abort();
                }

                if (l_a + l_b != count) {
                    std::cerr << "Error: sum mismatch\n";
                    std::cerr << "out of range results: nA=" << countA
                              << " nB=" << countB << " count=" << count
                              << " -> l_a=" << l_a << " l_b=" << l_b << "\n";
                    abort();
                }

                if (l_b > 0 && l_a < countA && !(A[l_a] > B[l_b - 1])) {
                    std::cerr << "Condition failed: A[l_a] > B[l_b-1]\n";
                    std::cout << A[l_a] << " > " << B[l_b - 1] << "\n";
                    std::cerr << "nA=" << countA << " nB=" << countB
                              << " count=" << count << " -> l_a="
                              << l_a << " l_b=" << l_b << "\n";
                    abort();
                }

                if (l_a > 0 && l_b < countB && !(B[l_b] > A[l_a - 1])) {
                    std::cerr << "Condition failed: B[l_b] > A[l_a - 1]\n";
                    std::cout << B[l_b] << " > " << A[l_a - 1] << "\n";
                    std::cerr << "nA=" << countA << " nB=" << countB
                              << " count=" << count << " -> l_a="
                              << l_a << " l_b=" << l_b << "\n";
                    abort();
                }
            }
        }
    }
}

template<typename value_t, typename index_t>
void test_partition3D_sweep(const index_t n) {
    for (index_t nA_exp = 0; nA_exp < n; nA_exp++) {
        const index_t countA = index_t(1ull << nA_exp);
        for (index_t nB_exp = 0; nB_exp < n; nB_exp++) {
            const index_t countB = index_t(1ull << nB_exp);
            for (index_t nC_exp = 0; nC_exp < n; nC_exp++) {
                const index_t countC = index_t(1ull << nC_exp);

                auto A = generate_sorted_vector<value_t>(countA);
                auto B = generate_sorted_vector<value_t>(countB);
                auto C = generate_sorted_vector<value_t>(countC);

                index_t max_count = countA + countB + countC;
                for (index_t count = 0; count < max_count; ++count) {
                    auto [l_a, l_b, l_c] = partition3D<value_t, index_t>(
                        A.data(), countA, B.data(), countB, C.data(), countC, count);

                    if (l_a < 0 || l_a > countA || l_b < 0 || l_b > countB || l_c < 0 || l_c > countC) {
                        std::cerr << "Out of range results: nA=" << countA
                                  << " nB=" << countB << " nC=" << countC << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }

                    if (l_a + l_b + l_c != count) {
                        std::cerr << "Error: sum mismatch\n";
                        std::cerr << "nA=" << countA << " nB=" << countB << " nC=" << countC
                                  << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }

                    if (l_b > 0 && l_a < countA && !(A[l_a] > B[l_b - 1])) {
                        std::cerr << "Condition failed: A[l_a] > B[l_b-1]\n";
                        std::cout << A[l_a] << " > " << B[l_b - 1] << "\n";
                        std::cerr << "nA=" << countA << " nB=" << countB << " nC=" << countC
                                  << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }

                    if (l_c > 0 && l_a < countA && !(A[l_a] > C[l_c - 1])) {
                        std::cerr << "Condition failed: A[l_a] > C[l_c-1]\n";
                        std::cout << A[l_a] << " > " << C[l_c - 1] << "\n";
                        std::cerr << "nA=" << countA << " nB=" << countB << " nC=" << countC
                                  << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }

                    if (l_a > 0 && l_b < countB && !(B[l_b] > A[l_a - 1])) {
                        std::cerr << "Condition failed: B[l_b] > A[l_a - 1]\n";
                        std::cout << B[l_b] << " > " << A[l_a - 1] << "\n";
                        std::cerr << "nA=" << countA << " nB=" << countB << " nC=" << countC
                                  << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }

                    if (l_c > 0 && l_b < countB && !(B[l_b] > C[l_c - 1])) {
                        std::cerr << "Condition failed: B[l_b] > C[l_c - 1]\n";
                        std::cout << B[l_b] << " > " << C[l_c - 1] << "\n";
                        std::cerr << "nA=" << countA << " nB=" << countB << " nC=" << countC
                                  << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }

                    if (l_a > 0 && l_c < countC && !(C[l_c] > A[l_a - 1])) {
                        std::cerr << "Condition failed: C[l_c] > A[l_a - 1]\n";
                        std::cout << C[l_c] << " > " << A[l_a - 1] << "\n";
                        std::cerr << "nA=" << countA << " nB=" << countB << " nC=" << countC
                                  << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }

                    if (l_b > 0 && l_c < countC && !(C[l_c] > B[l_b - 1])) {
                        std::cerr << "Condition failed: C[l_c] > B[l_b-1]\n";
                        std::cout << C[l_c] << " > " << B[l_b - 1] << "\n";
                        std::cerr << "nA=" << countA << " nB=" << countB << " nC=" << countC
                                  << " count=" << count
                                  << " -> l_a=" << l_a << " l_b=" << l_b << " l_c=" << l_c << "\n";
                        abort();
                    }
                }
            }
        }
    }
}

int main() {
    test_partition2D_sweep<double, int>(12);
    test_partition3D_sweep<double, int>(12);
    return 0;
}
