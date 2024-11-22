def karatsuba(x, y):
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    half = n // 2

    high_x = x // 10**half
    low_x = x % 10**half
    high_y = y // 10**half
    low_y = y % 10**half

    z0 = karatsuba(low_x, low_y)
    z1 = karatsuba((low_x + high_x), (low_y + high_y))
    z2 = karatsuba(high_x, high_y)

    return (z2 * 10**(2 * half)) + ((z1 - z2 - z0) * 10**half) + z0


def square_large_number(n):
    return karatsuba(n, n)

large_number = int(input("Enter a 20-digit number to square: "))
result = square_large_number(large_number)
print(f"Square of {large_number} is: {result}")

os2
def schedule_jobs(tasks):
    tasks.sort(key=lambda x: x[2], reverse=True)
    max_deadline = max(task[1] for task in tasks)
    result = [False] * max_deadline
    job_sequence = ['-1'] * max_deadline
    total_profit = 0

    for task in tasks:
        for j in range(min(max_deadline, task[1]) - 1, -1, -1):
            if not result[j]:
                result[j] = True
                job_sequence[j] = task[0]
                total_profit += task[2]
                break

    return job_sequence, total_profit

tasks = []
n = int(input("Enter the number of tasks: "))

for _ in range(n):
    task_name = input("Enter task name: ")
    deadline = int(input("Enter deadline: "))
    profit = int(input("Enter profit: "))
    tasks.append([task_name, deadline, profit])

scheduled_jobs, total_profit = schedule_jobs(tasks)

print("Scheduled Jobs:", scheduled_jobs)
print("Total Profit:", total_profit)

os3
def floyd_warshall(cost_matrix):
    num_offices = len(cost_matrix)
    # Initialize the distance matrix with infinity
    dist = [[float('inf')] * num_offices for _ in range(num_offices)]

    # Populate the initial distance matrix based on the input cost matrix
    for i in range(num_offices):
        for j in range(num_offices):
            if i == j:
                dist[i][j] = 0  # The cost to connect an office to itself is 0
            elif cost_matrix[i][j] != 0:
                dist[i][j] = cost_matrix[i][j]

    # Floyd-Warshall Algorithm to find shortest paths
    for k in range(num_offices):
        for i in range(num_offices):
            for j in range(num_offices):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

if __name__ == "__main__":
    num_offices = int(input("Enter the number of offices: "))
    cost_matrix = []
    print("Enter the cost matrix (enter 0 for no direct connection):")

    # Input the cost matrix
    for i in range(num_offices):
        row = list(map(int, input(f"Enter costs for office {i + 1} (space-separated): ").split()))
        cost_matrix.append(row)

    # Get the result from Floyd-Warshall
    result = floyd_warshall(cost_matrix)

    # Print the minimum connection costs between all offices
    print("Minimum connection costs between offices:")
    for row in result:
        print(row)
os4
import heapq

def dijkstra(N, edges, K):
    graph = [[] for _ in range(N + 1)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    dist = [float('inf')] * (N + 1)
    dist[K] = 0
    priority_queue = [(0, K)]

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        if current_dist > dist[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    max_distance = 0
    for d in dist[1:]:
        if d == float('inf'):
            return -1, dist[1:]
        max_distance = max(max_distance, d)

    return max_distance, dist[1:]

if __name__ == "__main__":
    try:
        N = int(input("Enter the number of nodes: "))
        M = int(input("Enter the number of edges: "))
        edges = []
        for _ in range(M):
            u, v, w = map(int, input("Enter edge (u, v, w) where u and v are nodes and w is the time: ").split())
            if 1 <= u <= N and 1 <= v <= N and w >= 0:
                edges.append((u, v, w))
            else:
                print("Invalid edge input. Ensure 1 <= u, v <= N and w >= 0.")

        K = int(input("Enter the starting node (K): "))
        if K < 1 or K > N:
            print("Invalid starting node. It should be between 1 and N.")
        else:
            result, distances = dijkstra(N, edges, K)
            print("Maximum distance:", result)
            print("Distances from node", K, ":", distances)
    except ValueError:
        print("Invalid input. Please enter integers only.")
os5
def is_valid_move(x, y, board, n):
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1

def knight_tour(n, start_x, start_y):
    board = [[-1 for _ in range(n)] for _ in range(n)]
    moves_x = [2, 1, -1, -2, -2, -1, 1, 2]
    moves_y = [1, 2, 2, 1, -1, -2, -2, -1]
    board[start_x][start_y] = 0

    if solve_knight_tour(start_x, start_y, 1, board, moves_x, moves_y, n):
        print("Knight's Tour solution:")
        for row in board:
            print(row)
    else:
        print("No solution exists.")

def solve_knight_tour(x, y, move_count, board, moves_x, moves_y, n):
    if move_count == n * n:
        return True

    for i in range(8):
        next_x = x + moves_x[i]
        next_y = y + moves_y[i]
        if is_valid_move(next_x, next_y, board, n):
            board[next_x][next_y] = move_count
            if solve_knight_tour(next_x, next_y, move_count + 1, board, moves_x, moves_y, n):
                return True
            board[next_x][next_y] = -1

    return False

try:
    n = int(input("Enter the size of the chessboard (NxN): "))
    if n < 1:
        print("Please enter a positive integer greater than 0.")
    else:
        start_x = int(input(f"Enter the starting x-coordinate (0 to {n-1}): "))
        start_y = int(input(f"Enter the starting y-coordinate (0 to {n-1}): "))
        
        if 0 <= start_x < n and 0 <= start_y < n:
            knight_tour(n, start_x, start_y)
        else:
            print("Invalid starting position!")
except ValueError:
    print("Invalid input! Please enter a valid integer.")
os6
def branch_and_bound(cost, N):
    min_cost = [float('inf')]
    optimal_assignment = [-1] * N
    assigned_clubs = [-1] * N
    visited = [False] * N

    def solve(level, current_cost):
        if level == N:
            if current_cost < min_cost[0]:
                min_cost[0] = current_cost
                # Copy the current assignment to optimal_assignment
                optimal_assignment[:] = assigned_clubs[:]
            return
        for i in range(N):
            if not visited[i]:
                visited[i] = True
                assigned_clubs[level] = i
                temp_cost = current_cost + cost[level][i]
                if temp_cost < min_cost[0]:
                    solve(level + 1, temp_cost)
                visited[i] = False
                assigned_clubs[level] = -1

    solve(0, 0)
    return min_cost[0], optimal_assignment

def get_user_input():
    N = int(input("Enter the number of students/clubs: "))
    cost = []
    print("Enter the cost matrix:")
    for i in range(N):
        row = list(map(int, input().strip().split()))
        cost.append(row)
    return cost, N

if __name__ == "__main__":
    cost_matrix, N = get_user_input()
    min_cost, optimal_assignment = branch_and_bound(cost_matrix, N)
    print("The minimum assignment cost is:", min_cost)
    print("Optimal assignment (Student to Club):", [x + 1 for x in optimal_assignment])  # Output club assignments as 1-indexed
