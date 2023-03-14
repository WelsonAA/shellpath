import array

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
from station import Station

givenPoints = np.array([[-171.60, 4.00],
                        [-206.40, 4.20],
                        [-255.90, 0.20],
                        [-272.10, -43.90],
                        [-205.50, -95.00],
                        [-185.50, -142.40],
                        [-151.10, -151.00],
                        [-101.40, -154.70],
                        [-47.80, -117.20],
                        [-43.80, -56.80],
                        [-43.90, -17.10],
                        [3.00, -2.70],
                        [47.80, -1.80],
                        [89.00, -5.50],
                        [45.90, -84.90],
                        [31.30, 19.30],
                        [36.30, 67.20],
                        [38.60, 155.10],
                        [74.00, 190.20],
                        [154.10, 177.30],
                        [189.20, 52.80],
                        [174.40, -148.00],
                        [10.20, -187.90],
                        [-145.80, -190.90],
                        [-232.60, 28.10],
                        [-119.40, 186.60],
                        [84.70, 144.10],
                        [148.10, 112.20],
                        [151.40, 15.20],
                        [124.70, 1.90],
                        [96.20, -28.60],
                        [-9.50, -88.30],
                        [-83.20, -87.70],
                        [-124.30, -42.40],
                        [-121.80, 28.10],
                        [-124.40, 106.30],
                        [-80.20, 133.30],
                        [-20.70, 87.90],
                        [25.70, 65.40],
                        [24.60, -30.70]])
intersections = np.array([
    [-229, -143],
    [-210, 156],
    [38, 187],
    [140, -123],
    [35, -141],
    [35, -92],
    [29, -3],
    [96, 72],
    [37, 91],
    [-44, 88],
    [-121, 85],
    [105, -3],
    [-40, 2],
    [-119, -4],
    [-121, -93],
    [-121, -147],
    [-120, 138],
    [154, -6],
    [-44, -89],
    [-244, -86],
    [-243, 82],
    [-186, 88],
    [-173, 137],
    [-186, 2],
    [-247, 76],
    [-184, -93],
    [35, -189],
    [92, -72]
])

edges = [[28, 29],
         [58, 29],
         [30, 58],
         [30, 52],
         [52, 14],
         [52, 31],
         [31, 68],
         [68, 15],
         [15, 46],
         [46, 32],
         [46, 45],
         [44, 45],
         [58, 44],
         [52, 48],
         [48, 49],
         [18, 49],
         [14, 13],
         [13, 47],
         [47, 16],
         [12, 47],
         [47, 40],
         [40, 46],
         [16, 39],
         [39, 17],
         [17, 49],
         [38, 49],
         [50, 38],
         [12, 53],
         [50, 37],
         [37, 57],
         [23, 67],
         [67, 45],
         [8, 9],
         [59, 9],
         [33, 59],
         [59, 10],
         [10, 11],
         [11, 53],
         [53, 50],
         [50, 51],
         [51, 36],
         [19, 43],
         [36, 57],
         [51, 35],
         [19, 20],
         [54, 53],
         [54, 35],
         [34, 55],
         [55, 33],
         [54, 1],
         [1, 64],
         [57, 63],
         [63, 62],
         [62, 64],
         [62, 65],
         [65, 3],
         [64, 2],
         [2, 3],
         [3, 4],
         [4, 60],
         [60, 5],
         [5, 66],
         [66, 55],
         [61, 25],
         [51, 62],
         [25, 41],
         [41, 24],
         [24, 23],
         [67, 22],
         [22, 21],
         [21, 21],
         [32, 59],
         [6, 66],
         [43, 18],
         [34, 54],
         [26, 43],
         [26, 42],
         [6, 7],
         [7, 56],
         [56, 8],
         [20, 21],
         [42, 61],
         [18, 27],
         [27, 28],
         [56, 55],
         [64, 66],
         [1,35],
         [1,34],
         [1,2],
         [4,5],
         [8,34],
         [7,34],
         [8,33],
         [9,33],
         [9,10],
         [32,33],
         [32,15],
         [32,40],
         [32,15],
         [15,31],
         [31,14],
         [31,30],
         [40,12],
         [40,13],
         [40,16],
         [12,13],
         [12,16],
         [13,16],[30,29],
         [32,23],
         [15,23],
         [24,25],
         [25,26]
         ]


def solve_tsp_aco(coords):
    # Get number of cities
    n = 40

    # Initialize parameters
    alpha = 4
    beta = 2
    rho = 0.5
    Q = 10
    max_iterations = 100
    best_distance = float('inf')
    best_route = None

    # Define the distance matrix
    """dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            dist[i][j] = d
            dist[j][i] = d"""

    # Initialize the pheromone trail matrix
    tau = np.ones((n, n))

    # Run the ACO algorithm for a fixed number of iterations
    for iteration in range(max_iterations):
        # Initialize the ants
        ants = np.zeros((n, n))
        for i in range(n):
            ants[i][0] = i

        # Loop over each ant
        for k in range(n):
            # Choose the starting city
            start_city = 0
            # Initialize the visited list and the route
            visited = np.zeros(n)
            visited[start_city] = 1
            route = [start_city]

            # Loop over the remaining cities
            #for i in range(n - 1):
            for i in range(st[start_city].neighbours.shape[0]):
                # Calculate the probabilities for the next city
                probs = np.zeros(n)
                denom = 0
                for j in range(n):
                    if visited[j] == 0:
                        numer = tau[start_city][j] ** alpha * (1.0 / graphMarix[start_city][j]) ** beta
                        probs[j] = numer
                        denom += numer
                probs = probs / denom

                # Choose the next city
                next_city = np.random.choice(range(st[start_city].neighbours.shape[0]), p=st[start_city].probs)
                visited[next_city] = 1
                route.append(next_city)
                ants[k][i + 1] = next_city
                start_city = next_city

            # Add the first city to the end of the route
            route.append(start_city)

            # Calculate the distance of the route
            distance = 0
            for i in range(n):
                distance += graphMarix[route[i]][route[(i + 1) % n]]

            # Update the best route
            if distance < best_distance:
                best_distance = distance
                best_route = route

        # Update the pheromone trail matrix
        delta_tau = np.zeros((n, n))
        for k in range(n):
            for i in range(n):
                delta_tau[int(ants[k][i])][int(ants[k][(i + 1) % n])] += Q / best_distance
        tau = (1 - rho) * tau + rho * delta_tau

    # Return the best route and distance
    return best_route, best_distance


"""route, distance = solve_tsp_aco(givenPoints)
for i in range(40):
    route[i] += 1"""


def printPath(start, end):
    print(start)
    if start == end:
        return
    start = next[start][end]
    printPath(start, end)


def get_point(s):
    s -= 1
    if s > 39:
        point = (intersections[s % 40][0], intersections[s % 40][1])
    else:
        point = (givenPoints[s][0], givenPoints[s][1])

    return point


TotalNodeCount = givenPoints.shape[0] + intersections.shape[0]

graphMarix = np.ones(shape=(TotalNodeCount + 1, TotalNodeCount + 1), dtype=float)
graphMarix = graphMarix * float('inf')

for s, d in edges:
    dis = math.dist(get_point(s), get_point(d))

    graphMarix[s][d] = dis
    graphMarix[d][s] = dis

np.fill_diagonal(graphMarix, 0)
graphMarix[0][0] = float('inf')
dp = graphMarix

next = np.ones(shape=(TotalNodeCount + 1, TotalNodeCount + 1), dtype=int)
next = next * -1

for j in range(1, TotalNodeCount + 1):
    for i in range(1, TotalNodeCount + 1):
        if dp[j][i] != float('inf'):
            next[j][i] = i

        # %%wljbcjdancb
for k in range(1, TotalNodeCount + 1):
    for j in range(1, TotalNodeCount + 1):
        for i in range(1, TotalNodeCount + 1):
            if dp[j][k] + dp[k][i] < dp[j][i]:
                dp[j][i] = dp[j][k] + dp[k][i]
                next[j][i] = next[j][k]

"""print("Best route:", route)
print("Distance:", distance)"""

st = []

for i in range(len(givenPoints)):
    st.append(Station(givenPoints[i][0], givenPoints[i][1]))
for edge in edges:
    if edge[0]>40 or edge[1]>40:
        continue
    else:
        st[edge[0] - 1].neighbours = np.append(st[edge[0] - 1].neighbours, edge[1] - 1)
        st[edge[0] - 1].probs=np.append(0)
        st[edge[1] - 1].neighbours = np.append(st[edge[1] - 1].neighbours, edge[0] - 1)
rt, dst = solve_tsp_aco(givenPoints)
for i in range(40):
    rt[i] += 1
print("Best route:", rt)
print("Distance:", dst)