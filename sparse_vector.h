template<typename index_t, typename value_t>
struct SparseVector {
    index_t length;
    index_t nnz;
    index_t *indices;
    value_t *values;
};


template<typename index_t, typename value_t>
index_t lower_bound(const SparseVector<index_t, value_t> a, index_t crd) {
    // First index >= crd
    index_t low = 0, high = a.nnz;
    while (low < high) {
        index_t mid = low + (high - low) / 2;
        if (a.indices[mid] < crd) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

template<typename index_t, typename value_t>
index_t upper_bound(const SparseVector<index_t, value_t> a, index_t crd) {
    // Last index <= crd
    index_t low = 0, high = a.nnz;
    while (low < high) {
        index_t mid = low + (high - low) / 2;
        if (a.indices[mid] <= crd) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low - 1;
}
