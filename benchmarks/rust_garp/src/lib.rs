use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Raw Floyd-Warshall GARP check in Rust.
///
/// No SCC decomposition — just the brute-force O(T^3) approach to measure
/// pure compute speed of Rust vs Python/Numba.
#[pyfunction]
fn check_garp_rust<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray2<'py, f64>,
    quantities: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
) -> PyResult<bool> {
    let p = prices.as_array();
    let q = quantities.as_array();
    let t = p.nrows();
    let n = p.ncols();

    // Build expenditure matrix E[i,j] = p_i @ x_j
    let mut e = Array2::<f64>::zeros((t, t));
    for i in 0..t {
        for j in 0..t {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += p[[i, k]] * q[[j, k]];
            }
            e[[i, j]] = sum;
        }
    }

    // Own expenditures
    let mut own_exp = vec![0.0f64; t];
    for i in 0..t {
        own_exp[i] = e[[i, i]];
    }

    // Build R (direct revealed preference) and P (strict)
    let mut r = vec![false; t * t];
    let mut p_strict = vec![false; t * t];

    for i in 0..t {
        for j in 0..t {
            r[i * t + j] = own_exp[i] >= e[[i, j]] - tolerance;
            p_strict[i * t + j] = own_exp[i] > e[[i, j]] + tolerance;
        }
        p_strict[i * t + i] = false; // No self-loops in P
    }

    // Floyd-Warshall transitive closure of R (in-place)
    // closure[i,j] = r[i*t+j]
    let mut closure = r.clone();
    // Set diagonal
    for i in 0..t {
        closure[i * t + i] = true;
    }

    for k in 0..t {
        for i in 0..t {
            if closure[i * t + k] {
                for j in 0..t {
                    if closure[k * t + j] {
                        closure[i * t + j] = true;
                    }
                }
            }
        }
    }

    // GARP violation check: R*[i,j] AND P[j,i]
    for i in 0..t {
        for j in 0..t {
            if closure[i * t + j] && p_strict[j * t + i] {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// GARP check with SCC optimization (matches Python implementation).
///
/// Uses Tarjan's SCC to decompose the graph, then only runs Floyd-Warshall
/// within each non-trivial SCC. Reachability across SCCs is propagated via
/// the condensed DAG.
#[pyfunction]
fn check_garp_rust_scc<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray2<'py, f64>,
    quantities: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
) -> PyResult<bool> {
    let p = prices.as_array();
    let q = quantities.as_array();
    let t = p.nrows();
    let n = p.ncols();

    // Build expenditure matrix
    let mut e = Array2::<f64>::zeros((t, t));
    for i in 0..t {
        for j in 0..t {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += p[[i, k]] * q[[j, k]];
            }
            e[[i, j]] = sum;
        }
    }

    let mut own_exp = vec![0.0f64; t];
    for i in 0..t {
        own_exp[i] = e[[i, i]];
    }

    // Build R and P
    let mut r_mat = vec![false; t * t];
    let mut p_strict = vec![false; t * t];
    for i in 0..t {
        for j in 0..t {
            r_mat[i * t + j] = own_exp[i] >= e[[i, j]] - tolerance;
            p_strict[i * t + j] = own_exp[i] > e[[i, j]] + tolerance;
        }
        p_strict[i * t + i] = false;
    }

    // Tarjan's SCC
    let scc_labels = tarjan_scc(&r_mat, t);
    let n_components = *scc_labels.iter().max().unwrap_or(&0) + 1;

    // Group nodes by SCC
    let mut scc_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    for (i, &label) in scc_labels.iter().enumerate() {
        scc_nodes[label].push(i);
    }

    // For each non-trivial SCC, run Floyd-Warshall and check violations
    let mut closure = r_mat.clone();
    for i in 0..t {
        closure[i * t + i] = true;
    }

    for scc in &scc_nodes {
        if scc.len() <= 1 {
            continue;
        }
        let k = scc.len();
        // Extract sub-adjacency and run FW on it
        let mut sub = vec![false; k * k];
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                sub[li * k + lj] = r_mat[ni * t + nj];
            }
            sub[li * k + li] = true;
        }
        // Floyd-Warshall on sub
        for kk in 0..k {
            for ii in 0..k {
                if sub[ii * k + kk] {
                    for jj in 0..k {
                        if sub[kk * k + jj] {
                            sub[ii * k + jj] = true;
                        }
                    }
                }
            }
        }
        // Write back
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                closure[ni * t + nj] = sub[li * k + lj];
            }
        }
    }

    // Propagate reachability across SCCs via condensed DAG
    // Build DAG
    let mut dag = vec![false; n_components * n_components];
    for i in 0..t {
        for j in 0..t {
            if r_mat[i * t + j] && scc_labels[i] != scc_labels[j] {
                dag[scc_labels[i] * n_components + scc_labels[j]] = true;
            }
        }
    }

    // Topological sort (Kahn's)
    let mut in_degree = vec![0usize; n_components];
    for i in 0..n_components {
        for j in 0..n_components {
            if dag[i * n_components + j] {
                in_degree[j] += 1;
            }
        }
    }
    let mut queue: Vec<usize> = (0..n_components).filter(|&i| in_degree[i] == 0).collect();
    let mut topo_order = Vec::with_capacity(n_components);
    let mut head = 0;
    while head < queue.len() {
        let node = queue[head];
        head += 1;
        topo_order.push(node);
        for j in 0..n_components {
            if dag[node * n_components + j] {
                in_degree[j] -= 1;
                if in_degree[j] == 0 {
                    queue.push(j);
                }
            }
        }
    }

    // Propagate reachability in reverse topological order
    // reachable[c] = set of all nodes reachable from SCC c
    let mut reachable: Vec<Vec<bool>> = (0..n_components)
        .map(|_| vec![false; t])
        .collect();

    // Initialize: each SCC reaches its own nodes (with internal TC)
    for c in 0..n_components {
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if closure[ni * t + j] {
                    reachable[c][j] = true;
                }
            }
        }
    }

    // Reverse topo order propagation
    for &c in topo_order.iter().rev() {
        for succ in 0..n_components {
            if dag[c * n_components + succ] {
                for j in 0..t {
                    if reachable[succ][j] {
                        reachable[c][j] = true;
                    }
                }
            }
        }
        // Write back to closure
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if reachable[c][j] {
                    closure[ni * t + j] = true;
                }
            }
        }
    }

    // GARP violation check
    for i in 0..t {
        for j in 0..t {
            if closure[i * t + j] && p_strict[j * t + i] {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Tarjan's SCC algorithm. Returns a vector of component labels.
fn tarjan_scc(adj: &[bool], n: usize) -> Vec<usize> {
    let mut index_counter = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![usize::MAX; n];
    let mut lowlinks = vec![0usize; n];
    let mut labels = vec![0usize; n];
    let mut current_label = 0usize;

    fn strongconnect(
        v: usize,
        n: usize,
        adj: &[bool],
        index_counter: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut Vec<bool>,
        indices: &mut Vec<usize>,
        lowlinks: &mut Vec<usize>,
        labels: &mut Vec<usize>,
        current_label: &mut usize,
    ) {
        indices[v] = *index_counter;
        lowlinks[v] = *index_counter;
        *index_counter += 1;
        stack.push(v);
        on_stack[v] = true;

        for w in 0..n {
            if adj[v * n + w] {
                if indices[w] == usize::MAX {
                    strongconnect(
                        w, n, adj, index_counter, stack, on_stack, indices, lowlinks,
                        labels, current_label,
                    );
                    lowlinks[v] = lowlinks[v].min(lowlinks[w]);
                } else if on_stack[w] {
                    lowlinks[v] = lowlinks[v].min(indices[w]);
                }
            }
        }

        if lowlinks[v] == indices[v] {
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                labels[w] = *current_label;
                if w == v {
                    break;
                }
            }
            *current_label += 1;
        }
    }

    for v in 0..n {
        if indices[v] == usize::MAX {
            strongconnect(
                v, n, adj, &mut index_counter, &mut stack, &mut on_stack,
                &mut indices, &mut lowlinks, &mut labels, &mut current_label,
            );
        }
    }

    labels
}

/// Parallel Floyd-Warshall GARP check using Rayon.
///
/// Parallelizes the row updates in the FW inner loop, matching
/// Python/Numba's prange strategy for fair comparison.
#[pyfunction]
fn check_garp_rust_parallel<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray2<'py, f64>,
    quantities: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
) -> PyResult<bool> {
    let p = prices.as_array();
    let q = quantities.as_array();
    let t = p.nrows();
    let n = p.ncols();

    // Build expenditure matrix
    let mut e_flat = vec![0.0f64; t * t];
    for i in 0..t {
        for j in 0..t {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += p[[i, k]] * q[[j, k]];
            }
            e_flat[i * t + j] = sum;
        }
    }

    let mut own_exp = vec![0.0f64; t];
    for i in 0..t {
        own_exp[i] = e_flat[i * t + i];
    }

    // Build R and P
    let mut r_mat = vec![false; t * t];
    let mut p_strict = vec![false; t * t];
    for i in 0..t {
        for j in 0..t {
            r_mat[i * t + j] = own_exp[i] >= e_flat[i * t + j] - tolerance;
            p_strict[i * t + j] = own_exp[i] > e_flat[i * t + j] + tolerance;
        }
        p_strict[i * t + i] = false;
    }

    // Parallel Floyd-Warshall using Rayon
    let mut closure = r_mat.clone();
    for i in 0..t {
        closure[i * t + i] = true;
    }

    for k in 0..t {
        // Extract col_k and row_k for this iteration
        let col_k: Vec<bool> = (0..t).map(|i| closure[i * t + k]).collect();
        let row_k: Vec<bool> = (0..t).map(|j| closure[k * t + j]).collect();

        // Parallel update over rows
        closure
            .par_chunks_mut(t)
            .enumerate()
            .for_each(|(i, row)| {
                if col_k[i] {
                    for j in 0..t {
                        if row_k[j] {
                            row[j] = true;
                        }
                    }
                }
            });
    }

    // GARP violation check
    for i in 0..t {
        for j in 0..t {
            if closure[i * t + j] && p_strict[j * t + i] {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Original batch function (kept for backward compatibility / comparison).
#[pyfunction]
fn check_garp_batch_rust<'py>(
    _py: Python<'py>,
    prices_list: Vec<PyReadonlyArray2<'py, f64>>,
    quantities_list: Vec<PyReadonlyArray2<'py, f64>>,
    tolerance: f64,
) -> PyResult<Vec<bool>> {
    let n_users = prices_list.len();

    let users: Vec<(Vec<f64>, Vec<f64>, usize, usize)> = (0..n_users)
        .map(|i| {
            let p = prices_list[i].as_array();
            let q = quantities_list[i].as_array();
            let t = p.nrows();
            let n = p.ncols();
            let p_flat: Vec<f64> = p.iter().cloned().collect();
            let q_flat: Vec<f64> = q.iter().cloned().collect();
            (p_flat, q_flat, t, n)
        })
        .collect();

    let results: Vec<bool> = users
        .par_iter()
        .map(|(p_flat, q_flat, t, n)| {
            garp_check_flat(p_flat, q_flat, *t, *n, tolerance)
        })
        .collect();

    Ok(results)
}

/// Memory-optimized batch: thread-local scratchpads + bit-packed closure.
///
/// Instead of allocating fresh Vec<bool> per user (10M allocations),
/// each Rayon thread reuses a single pre-allocated scratchpad.
/// The closure matrix uses FixedBitSet (1 bit per entry) instead of
/// Vec<bool> (1 byte per entry) — 8x memory reduction + better cache.
#[pyfunction]
fn check_garp_batch_optimized<'py>(
    _py: Python<'py>,
    prices_list: Vec<PyReadonlyArray2<'py, f64>>,
    quantities_list: Vec<PyReadonlyArray2<'py, f64>>,
    tolerance: f64,
) -> PyResult<Vec<bool>> {
    let n_users = prices_list.len();

    // Copy user data out of Python (required for Send across threads)
    let users: Vec<(Vec<f64>, Vec<f64>, usize, usize)> = (0..n_users)
        .map(|i| {
            let p = prices_list[i].as_array();
            let q = quantities_list[i].as_array();
            let t = p.nrows();
            let n = p.ncols();
            (p.iter().cloned().collect(), q.iter().cloned().collect(), t, n)
        })
        .collect();

    // Find max T across all users to size scratchpads
    let max_t = users.iter().map(|(_, _, t, _)| *t).max().unwrap_or(0);

    // Rayon with thread-local scratchpads via map_init
    let results: Vec<bool> = users
        .par_iter()
        .map_init(
            || Scratchpad::new(max_t),
            |scratch, (p_flat, q_flat, t, n)| {
                garp_check_with_scratchpad(scratch, p_flat, q_flat, *t, *n, tolerance)
            },
        )
        .collect();

    Ok(results)
}

/// Thread-local reusable buffers. One per Rayon thread.
/// Avoids millions of heap allocations.
struct Scratchpad {
    e: Vec<f64>,           // Expenditure matrix (T x T)
    own_exp: Vec<f64>,     // Diagonal (T)
    closure: FixedBitSet,  // Bit-packed transitive closure (T x T bits)
    p_strict: FixedBitSet, // Bit-packed strict preference (T x T bits)
    r_mat: FixedBitSet,    // Bit-packed R matrix (T x T bits)
    capacity: usize,       // Max T this scratchpad supports
}

use fixedbitset::FixedBitSet;

impl Scratchpad {
    fn new(max_t: usize) -> Self {
        let n2 = max_t * max_t;
        Scratchpad {
            e: vec![0.0; n2],
            own_exp: vec![0.0; max_t],
            closure: FixedBitSet::with_capacity(n2),
            p_strict: FixedBitSet::with_capacity(n2),
            r_mat: FixedBitSet::with_capacity(n2),
            capacity: max_t,
        }
    }

    fn ensure_capacity(&mut self, t: usize) {
        if t > self.capacity {
            let n2 = t * t;
            self.e.resize(n2, 0.0);
            self.own_exp.resize(t, 0.0);
            self.closure.grow(n2);
            self.p_strict.grow(n2);
            self.r_mat.grow(n2);
            self.capacity = t;
        }
    }
}

fn garp_check_with_scratchpad(
    scratch: &mut Scratchpad,
    p_flat: &[f64],
    q_flat: &[f64],
    t: usize,
    n: usize,
    tolerance: f64,
) -> bool {
    scratch.ensure_capacity(t);
    let tt = t * t;

    // Build expenditure matrix into scratchpad
    for i in 0..t {
        for j in 0..t {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += p_flat[i * n + k] * q_flat[j * n + k];
            }
            scratch.e[i * t + j] = sum;
        }
    }

    for i in 0..t {
        scratch.own_exp[i] = scratch.e[i * t + i];
    }

    // Build R and P into bit-packed buffers
    scratch.r_mat.clear();
    scratch.p_strict.clear();
    scratch.closure.clear();

    for i in 0..t {
        for j in 0..t {
            let idx = i * t + j;
            if scratch.own_exp[i] >= scratch.e[idx] - tolerance {
                scratch.r_mat.insert(idx);
                scratch.closure.insert(idx);
            }
            if i != j && scratch.own_exp[i] > scratch.e[idx] + tolerance {
                scratch.p_strict.insert(idx);
            }
        }
        // Self-loop in closure
        scratch.closure.insert(i * t + i);
    }

    // SCC-based transitive closure (matches Python production code)
    // Use Vec<bool> for the FW inner loop (faster than bitset for dense access)
    let mut r_bool = vec![false; t * t];
    let mut closure_bool = vec![false; t * t];
    let mut p_strict_bool = vec![false; t * t];
    for i in 0..t {
        for j in 0..t {
            let idx = i * t + j;
            r_bool[idx] = scratch.r_mat.contains(idx);
            closure_bool[idx] = r_bool[idx];
            p_strict_bool[idx] = scratch.p_strict.contains(idx);
        }
        closure_bool[i * t + i] = true;
    }

    // Find SCCs
    let scc_labels = tarjan_scc(&r_bool, t);
    let n_components = scc_labels.iter().cloned().max().unwrap_or(0) + 1;

    let mut scc_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    for (i, &label) in scc_labels.iter().enumerate() {
        scc_nodes[label].push(i);
    }

    // FW only within non-trivial SCCs (Vec<bool> for speed)
    for scc in &scc_nodes {
        if scc.len() <= 1 { continue; }
        let k = scc.len();
        let mut sub = vec![false; k * k];
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                sub[li * k + lj] = closure_bool[ni * t + nj];
            }
            sub[li * k + li] = true;
        }
        for kk in 0..k {
            for ii in 0..k {
                if sub[ii * k + kk] {
                    for jj in 0..k {
                        if sub[kk * k + jj] { sub[ii * k + jj] = true; }
                    }
                }
            }
        }
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                closure_bool[ni * t + nj] = sub[li * k + lj];
            }
        }
    }

    // DAG reachability propagation
    let mut dag = vec![false; n_components * n_components];
    for i in 0..t {
        for j in 0..t {
            if r_bool[i * t + j] && scc_labels[i] != scc_labels[j] {
                dag[scc_labels[i] * n_components + scc_labels[j]] = true;
            }
        }
    }

    // Topo sort
    let mut in_deg = vec![0usize; n_components];
    for i in 0..n_components {
        for j in 0..n_components {
            if dag[i * n_components + j] { in_deg[j] += 1; }
        }
    }
    let mut queue: Vec<usize> = (0..n_components).filter(|&i| in_deg[i] == 0).collect();
    let mut topo = Vec::with_capacity(n_components);
    let mut head = 0;
    while head < queue.len() {
        let nd = queue[head]; head += 1; topo.push(nd);
        for j in 0..n_components {
            if dag[nd * n_components + j] { in_deg[j] -= 1; if in_deg[j] == 0 { queue.push(j); } }
        }
    }

    // Propagate (using bool vectors for speed)
    let mut reach: Vec<Vec<bool>> = (0..n_components).map(|_| vec![false; t]).collect();
    for c in 0..n_components {
        for &ni in &scc_nodes[c] {
            for j in 0..t { if closure_bool[ni * t + j] { reach[c][j] = true; } }
        }
    }
    for &c in topo.iter().rev() {
        for succ in 0..n_components {
            if dag[c * n_components + succ] {
                for j in 0..t { if reach[succ][j] { reach[c][j] = true; } }
            }
        }
        for &ni in &scc_nodes[c] {
            for j in 0..t { if reach[c][j] { closure_bool[ni * t + j] = true; } }
        }
    }

    // Violation check
    for i in 0..t {
        for j in 0..t {
            if closure_bool[i * t + j] && p_strict_bool[j * t + i] { return false; }
        }
    }

    true
}

/// Original flat check (no scratchpad, no bitpacking).
fn garp_check_flat(p_flat: &[f64], q_flat: &[f64], t: usize, n: usize, tolerance: f64) -> bool {
    let mut e = vec![0.0f64; t * t];
    for i in 0..t {
        for j in 0..t {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += p_flat[i * n + k] * q_flat[j * n + k];
            }
            e[i * t + j] = sum;
        }
    }

    let mut own_exp = vec![0.0f64; t];
    for i in 0..t {
        own_exp[i] = e[i * t + i];
    }

    let mut r = vec![false; t * t];
    let mut p_strict = vec![false; t * t];
    for i in 0..t {
        for j in 0..t {
            r[i * t + j] = own_exp[i] >= e[i * t + j] - tolerance;
            p_strict[i * t + j] = own_exp[i] > e[i * t + j] + tolerance;
        }
        p_strict[i * t + i] = false;
    }

    let mut closure = r;
    for i in 0..t {
        closure[i * t + i] = true;
    }
    for k in 0..t {
        for i in 0..t {
            if closure[i * t + k] {
                for j in 0..t {
                    if closure[k * t + j] {
                        closure[i * t + j] = true;
                    }
                }
            }
        }
    }

    for i in 0..t {
        for j in 0..t {
            if closure[i * t + j] && p_strict[j * t + i] {
                return false;
            }
        }
    }

    true
}

#[pymodule]
fn rust_garp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_garp_rust, m)?)?;
    m.add_function(wrap_pyfunction!(check_garp_rust_scc, m)?)?;
    m.add_function(wrap_pyfunction!(check_garp_rust_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(check_garp_batch_rust, m)?)?;
    m.add_function(wrap_pyfunction!(check_garp_batch_optimized, m)?)?;
    Ok(())
}
