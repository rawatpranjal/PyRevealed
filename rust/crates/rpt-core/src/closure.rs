use crate::scc::tarjan_scc;

/// SCC-optimized transitive closure.
///
/// 1. Find SCCs via Tarjan's (O(V+E))
/// 2. Run Floyd-Warshall only within each non-trivial SCC
/// 3. Propagate reachability across the condensed DAG
///
/// Returns: n_components and max_scc_size (for diagnostics).
/// The closure is written in-place into the provided buffer.
pub fn scc_transitive_closure(
    r_mat: &[bool],
    t: usize,
    closure: &mut [bool],
    scc_labels: &mut [u32],
) -> (usize, usize) {
    // Initialize closure = R + diagonal
    closure[..t * t].copy_from_slice(&r_mat[..t * t]);
    for i in 0..t {
        closure[i * t + i] = true;
    }

    // Find SCCs
    let n_comp = tarjan_scc(r_mat, t, scc_labels);

    if n_comp <= 1 && t > 1 {
        // Single SCC - run full Floyd-Warshall
        floyd_warshall(closure, t);
        return (1, t);
    }

    // Group nodes by SCC, find max SCC size
    let mut scc_sizes = vec![0usize; n_comp];
    for i in 0..t {
        scc_sizes[scc_labels[i] as usize] += 1;
    }
    let max_scc = *scc_sizes.iter().max().unwrap_or(&0);

    // Collect SCC node lists
    let mut scc_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for i in 0..t {
        scc_nodes[scc_labels[i] as usize].push(i);
    }

    // Floyd-Warshall within each non-trivial SCC
    for scc in &scc_nodes {
        if scc.len() <= 1 {
            continue;
        }
        let k = scc.len();
        let mut sub = vec![false; k * k];
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                sub[li * k + lj] = closure[ni * t + nj];
            }
            sub[li * k + li] = true;
        }
        floyd_warshall(&mut sub, k);
        for (li, &ni) in scc.iter().enumerate() {
            for (lj, &nj) in scc.iter().enumerate() {
                closure[ni * t + nj] = sub[li * k + lj];
            }
        }
    }

    // Build condensed DAG
    let mut dag = vec![false; n_comp * n_comp];
    for i in 0..t {
        for j in 0..t {
            let si = scc_labels[i] as usize;
            let sj = scc_labels[j] as usize;
            if r_mat[i * t + j] && si != sj {
                dag[si * n_comp + sj] = true;
            }
        }
    }

    // Topological sort (Kahn's)
    let mut in_deg = vec![0usize; n_comp];
    for i in 0..n_comp {
        for j in 0..n_comp {
            if dag[i * n_comp + j] {
                in_deg[j] += 1;
            }
        }
    }
    let mut queue: Vec<usize> = (0..n_comp).filter(|&i| in_deg[i] == 0).collect();
    let mut topo = Vec::with_capacity(n_comp);
    let mut head = 0;
    while head < queue.len() {
        let nd = queue[head];
        head += 1;
        topo.push(nd);
        for j in 0..n_comp {
            if dag[nd * n_comp + j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 {
                    queue.push(j);
                }
            }
        }
    }

    // Propagate reachability in reverse topological order
    let mut reach: Vec<Vec<bool>> = (0..n_comp).map(|_| vec![false; t]).collect();
    for c in 0..n_comp {
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if closure[ni * t + j] {
                    reach[c][j] = true;
                }
            }
        }
    }
    for &c in topo.iter().rev() {
        for succ in 0..n_comp {
            if dag[c * n_comp + succ] {
                for j in 0..t {
                    if reach[succ][j] {
                        reach[c][j] = true;
                    }
                }
            }
        }
        for &ni in &scc_nodes[c] {
            for j in 0..t {
                if reach[c][j] {
                    closure[ni * t + j] = true;
                }
            }
        }
    }

    (n_comp, max_scc)
}

/// Raw Floyd-Warshall transitive closure on a flat bool matrix.
#[inline]
fn floyd_warshall(closure: &mut [bool], t: usize) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_reachability() {
        // 0->1->2
        let mut r = vec![false; 9];
        r[0 * 3 + 1] = true;
        r[1 * 3 + 2] = true;
        let mut closure = vec![false; 9];
        let mut labels = [0u32; 3];
        scc_transitive_closure(&r, 3, &mut closure, &mut labels);
        assert!(closure[0 * 3 + 2]); // 0 reaches 2
        assert!(!closure[2 * 3 + 0]); // 2 does NOT reach 0
    }

    #[test]
    fn test_cycle_all_reach_all() {
        // 0->1->2->0
        let mut r = vec![false; 9];
        r[0 * 3 + 1] = true;
        r[1 * 3 + 2] = true;
        r[2 * 3 + 0] = true;
        let mut closure = vec![false; 9];
        let mut labels = [0u32; 3];
        scc_transitive_closure(&r, 3, &mut closure, &mut labels);
        // All reach all
        for i in 0..3 {
            for j in 0..3 {
                assert!(closure[i * 3 + j], "expected {i}->{j} reachable");
            }
        }
    }
}
