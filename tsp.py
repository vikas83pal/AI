import itertools

distance = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

n = len(distance)

cities = list(range(n))

min_path = None
min_cost = float('inf')

for prem in itertools.permutations(cities[1:]):
    path = [0] + list(prem) + [0]
    cost = 0

    for i in range(len(path) - 1):
        cost += distance[path[i]][path[i + 1]]

        if cost < min_cost:
            min_cost = cost
            min_path = path

print("Shortest path is : ", min_path)
print("Minimum cost is : ", min_cost)