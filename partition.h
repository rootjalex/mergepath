#pragma once
#include <tuple>

// Find l_a and l_b
// such that
// l_a + l_b == count
// if l_b > 0: A[l_a] > B[l_b - 1]
// if l_a > 0: B[l_b] > A[l_a - 1]
template<typename value_t, typename index_t>
std::tuple<index_t, index_t> partition2D(const value_t *A, const index_t nA,
                                     const value_t *B, const index_t nB,
                                     const index_t count) {
    // Binary search over l_a, l_b always is count - l_a
    index_t low = std::max<index_t>(0, (int64_t)count - (int64_t)nB);
    index_t high = std::min<index_t>(count, nA);

    while (low < high) {
        index_t l_a = low + (high - low) / 2;
        index_t l_b = count - l_a;

        if (l_b > 0 && l_a < nA && B[l_b - 1] > A[l_a]) {
            // l_a too small, must move right to increase l_a
            low = l_a + 1;
        } else if (l_a > 0 && l_b < nB && A[l_a - 1] > B[l_b]) {
            // l_a too big, must move left to decrease l_a
            high = l_a;
        } else {
            // Boundary conditions hit, or found point.
            low = l_a;
            break;
        }
    }

    index_t l_a = low;
    index_t l_b = count - l_a;
    return std::make_tuple(l_a, l_b);
}

// Find l_a, l_b, and l_c
// such that
// l_a + l_b + l_c == count
// if l_b > 0: A[l_a] > B[l_b - 1]
// if l_c > 0: A[l_a] > C[l_c - 1]
// if l_a > 0: B[l_b] > A[l_a - 1]
// if l_c > 0: B[l_b] > C[l_c - 1]
// if l_a > 0: C[l_c] > A[l_a - 1]
// if l_b > 0: C[l_c] > B[l_b - 1]
template<typename value_t, typename index_t>
std::tuple<index_t, index_t, index_t> partition3D(const value_t *A, const index_t nA,
                                     const value_t *B, const index_t nB,
                                     const value_t *C, const index_t nC,
                                     const index_t count) {
    // Binary search over l_a, then over B/C
    index_t low = std::max<index_t>(0, count > (nB + nC) ? count - (nB + nC) : 0);
    index_t high = std::min<index_t>(count, nA);

    while (low < high) {
        index_t l_a = low + (high - low) / 2;
        index_t remaining = count - l_a;

        // Binary search over B/C
        auto [l_b, l_c] = partition2D(B, nB, C, nC, remaining);

        // Get (clamped) predecessors
        value_t a_left = (l_a > 0) ? A[l_a - 1] : std::numeric_limits<value_t>::lowest();
        value_t b_left = (l_b > 0) ? B[l_b - 1] : std::numeric_limits<value_t>::lowest();
        value_t c_left = (l_c > 0) ? C[l_c - 1] : std::numeric_limits<value_t>::lowest();

        // Get (clamped) values
        value_t a = (l_a < nA) ? A[l_a] : std::numeric_limits<value_t>::max();
        value_t b = (l_b < nB) ? B[l_b] : std::numeric_limits<value_t>::max();
        value_t c = (l_c < nC) ? C[l_c] : std::numeric_limits<value_t>::max();

        bool valid =
            (l_b == 0 || a > b_left) &&
            (l_c == 0 || a > c_left) &&
            (l_a == 0 || b > a_left) &&
            (l_c == 0 || b > c_left) &&
            (l_a == 0 || c > a_left) &&
            (l_b == 0 || c > b_left);

        if (valid) {
            return std::make_tuple(l_a, l_b, l_c);
        }

        if (a <= std::max<value_t>(b_left, c_left)) {
            // a is too small, move right
            low = l_a + 1;
        } else {
            // a is too big, move left
            high = l_a;
        }
    }

    index_t l_a = low;
    auto [l_b, l_c] = partition2D<value_t, index_t>(B, nB, C, nC, count - l_a);
    return std::make_tuple(l_a, l_b, l_c);
}