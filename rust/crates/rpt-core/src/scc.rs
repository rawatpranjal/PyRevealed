/// Tarjan's Strongly Connected Components algorithm.
///
/// Finds all SCCs of a directed graph in O(V + E) time.
/// Returns a vector of component labels (one per node) and the number of components.
pub fn tarjan_scc(adj: &[bool], n: usize, labels: &mut [u32]) -> usize {
    let mut index_counter = 0usize;
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut on_stack = vec![false; n];
    let mut indices = vec![usize::MAX; n];
    let mut lowlinks = vec![0usize; n];
    let mut current_label = 0usize;

    for i in 0..n {
        labels[i] = 0;
    }

    fn strongconnect(
        v: usize,
        n: usize,
        adj: &[bool],
        index_counter: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut Vec<bool>,
        indices: &mut Vec<usize>,
        lowlinks: &mut Vec<usize>,
        labels: &mut [u32],
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
                labels[w] = *current_label as u32;
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
                &mut indices, &mut lowlinks, labels, &mut current_label,
            );
        }
    }

    current_label
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_scc() {
        // 3-cycle: 0->1->2->0
        let mut adj = vec![false; 9];
        adj[0 * 3 + 1] = true;
        adj[1 * 3 + 2] = true;
        adj[2 * 3 + 0] = true;
        let mut labels = [0u32; 3];
        let n = tarjan_scc(&adj, 3, &mut labels);
        assert_eq!(n, 1);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_two_sccs() {
        // 0->1, 1->0 (SCC A), 2->3, 3->2 (SCC B), 1->2 (bridge)
        let mut adj = vec![false; 16];
        adj[0 * 4 + 1] = true;
        adj[1 * 4 + 0] = true;
        adj[2 * 4 + 3] = true;
        adj[3 * 4 + 2] = true;
        adj[1 * 4 + 2] = true;
        let mut labels = [0u32; 4];
        let n = tarjan_scc(&adj, 4, &mut labels);
        assert_eq!(n, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_all_singletons() {
        // DAG: 0->1->2
        let mut adj = vec![false; 9];
        adj[0 * 3 + 1] = true;
        adj[1 * 3 + 2] = true;
        let mut labels = [0u32; 3];
        let n = tarjan_scc(&adj, 3, &mut labels);
        assert_eq!(n, 3);
    }
}
