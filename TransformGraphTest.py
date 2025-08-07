import algorithms as gt

def main():
    """
    Main function to set up a test graph, run the transformation,
    and print the results for verification.
    """
    print("--- Graph Transformation Test ---")

    # --- 1. Setup a small, simple directed graph for testing ---
    # We will use a 4-node directed line graph: 0 -> 1 -> 2 -> 3
    # This makes the transformation easy to trace.
    gt.N = 4
    gt.M = 0
    gt.start = 0 # Start from node 0
    gt.adj = [set() for _ in range(gt.N)]

    # Define the directed edges and their weights
    edges = [(0, 1, 5), (1, 2, 10), (2, 3, 15), (3, 1, 1)]
    for u, v, w in edges:
        gt.adj[u].add((v, w))
        gt.M += 1 # Each edge is now directed

    # --- 2. Print the state of the original graph ---
    print("\n--- Original Graph State ---")
    print(f"Number of vertices (N): {gt.N}")
    print(f"Number of edges (M): {gt.M}")
    print(f"Start node: {gt.start}")
    print("Adjacency List (adj):")
    for i in range(gt.N):
        print(f"  Node {i}: {gt.adj[i]}")

    # --- 3. Run the graph transformation function ---
    print("\n... Transforming graph ...")
    gt.transformGraph()
    print("... Transformation complete! ...")

    # --- 4. Print the state of the transformed graph ---
    print("\n--- Transformed Graph State ---")
    print(f"New number of vertices (N): {gt.N}")
    print(f"New number of edges (M): {gt.M}")
    print(f"New start node: {gt.start}")
    print("New Adjacency List (adj):")
    # Only print the first few nodes for brevity if the graph is large
    nodes_to_print = min(gt.N, 20)
    for i in range(nodes_to_print):
        print(f"  New Node {i}: {gt.adj[i]}")
    if gt.N > nodes_to_print:
        print(f"  ... (and {gt.N - nodes_to_print} more nodes)")

    print("\n--- Test Finished ---")


if __name__ == "__main__":
    # This ensures the main function is called only when the script is executed directly.
    main()