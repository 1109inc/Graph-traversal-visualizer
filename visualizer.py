import pygame
import sys
import math
import tracemalloc
import time
import heapq
from collections import deque

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 1500, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Node-Based Traversal Visualizer")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# Node and Edge Data
nodes = []
edges = []
weights = {}
start_node = None
end_node = None
traversal_type = None
adding_edges = False
edges_fixed = False
first_node = None
instruction = "Press anywhere to add nodes, press Enter to proceed"
time_taken = None
weighted_graph = None
is_directed=None
euclidean=None
waiting_for_heuristic=None
curr=None
peak=None
skip_next_click = False
RESET_BTN_RECT = pygame.Rect(20, height - 50, 100, 30)
# Draw the nodes, edges, instructions, and time taken
def draw_graph():
    
    global time_taken
    screen.fill(WHITE)
    def draw_arrowhead(start, end, radius=15, color=BLACK, size=15):
        import math

        # Calculate the direction of the edge
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx)

        # Shorten the end point by 'radius' to touch the edge of the circle
        tip_x = end[0] - radius * math.cos(angle)
        tip_y = end[1] - radius * math.sin(angle)

        # Calculate the arrowhead points
        arrow_angle = math.pi / 6  # 30 degrees
        x1 = tip_x - size * math.cos(angle - arrow_angle)
        y1 = tip_y - size * math.sin(angle - arrow_angle)
        x2 = tip_x - size * math.cos(angle + arrow_angle)
        y2 = tip_y - size * math.sin(angle + arrow_angle)

        # Draw the arrowhead as a triangle
        pygame.draw.polygon(screen, color, [(tip_x, tip_y), (x1, y1), (x2, y2)])

    for edge in edges:
        start_pos = nodes[edge[0]]
        end_pos = nodes[edge[1]]
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        if is_directed:
            # Draw arrowhead
            draw_arrowhead(start_pos, end_pos)
        if weighted_graph:
            font = pygame.font.SysFont(None, 20)
    
            # Midpoint of edge
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
    
            # Calculate angle and perpendicular offset
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            angle = math.atan2(dy, dx)
    
            offset = 15  # distance above the edge
            perp_angle = angle + math.pi / 2
            label_x = mid_x - offset * math.cos(perp_angle)
            label_y = mid_y - offset * math.sin(perp_angle)

            # Get weight
            weight = weights.get(edge) or weights.get((edge[1], edge[0]))
            if weight is not None:
                angle_deg = -math.degrees(angle)
                if angle_deg < -90 or angle_deg > 90:
                    angle_deg += 180
                text_surface = font.render(str(weight), True, BLACK)
                text_surface = pygame.transform.rotate(text_surface, angle_deg)
                text_rect = text_surface.get_rect(center=(label_x, label_y))
                screen.blit(text_surface, text_rect)
            
    for i, node in enumerate(nodes):
        if i == start_node:
            color = GREEN
        elif i == end_node:
            color = RED
        else:
            color = '#66ffff'
        pygame.draw.circle(screen, color, node, 15)
        font = pygame.font.SysFont(None, 24)
        text = font.render(str(i + 1), True, BLACK)
        text_rect = text.get_rect(center=node)
        screen.blit(text, text_rect)
    font = pygame.font.SysFont(None, 32)
    text = font.render(instruction, True, BLACK)
    screen.blit(text, (20, 20))
    if time_taken is not None:
        time_text = font.render(f"Time Taken: {time_taken*1e6:.2f} microseconds", True, RED)
        screen.blit(time_text, (20, 60))
    if curr is not None:
        memory_current=font.render(f"Current Memory Usage: {curr/1024:0.2f} KB",True,RED)
        screen.blit(memory_current, (20, 90))
    if peak is not None:
        memory_peak=font.render(f"Peak Memory Usage: {peak/1024:0.2f} KB",True,RED)
        screen.blit(memory_peak, (20, 120))
    pygame.draw.rect(screen, RED, RESET_BTN_RECT)
    font = pygame.font.SysFont(None, 24)
    reset_text = font.render("Reset", True, BLACK)
    text_rect = reset_text.get_rect(center=RESET_BTN_RECT.center)
    screen.blit(reset_text, text_rect)
    pygame.display.flip()

# DFS Traversal
def dfs(start):
    visited = set()
    stack = [start]
    path = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            path.append(node)
            for i, j in sorted(edges):
                if i == node and j not in visited:
                    stack.append(j)
                elif j == node and i not in visited and not is_directed:
                    stack.append(i)
    return path

# BFS Traversal
def bfs(start):
    visited = set([start])
    queue = deque([start])
    path = []
    while queue:
        node = queue.popleft()
        path.append(node)
        neighbors = []
        for i, j in edges:
            if i == node:
                neighbors.append(j)
            elif j == node and not is_directed:
                neighbors.append(i)
        for neighbor in sorted(neighbors):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return path

# Dijkstra's Algorithm
def dijkstra(start, end):
    dist = {i: float('inf') for i in range(len(nodes))}
    dist[start] = 0
    prev = {i: None for i in range(len(nodes))}
    pq = [(0, start)]

    graph = {i: [] for i in range(len(nodes))}
    for u, v in edges:
        w = weights.get((u, v), weights.get((v, u), 1))
        graph[u].append((v, w))
        if not is_directed:
            graph[v].append((u, w))

    while pq:
        current_dist, u = heapq.heappop(pq)
        if u == end:
            break
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1] if path[-1] == start else []

# A* Algorithm
def heuristiceuclidean(a, b):
    return math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
def heuristicmanhatten(a,b):
    return abs(nodes[a][0] - nodes[b][0]) + abs(nodes[a][1] - nodes[b][1])
def a_star(start, end):
    open_set = [(0, start)]
    came_from = {}
    g_score = {i: float('inf') for i in range(len(nodes))}
    f_score = {i: float('inf') for i in range(len(nodes))}
    g_score[start] = 0
    f_score[start] = heuristiceuclidean(start, end) if euclidean else heuristicmanhatten(start, end)
    visited = set()

    graph = {i: [] for i in range(len(nodes))}
    for u, v in edges:
        w = weights.get((u, v), weights.get((v, u), 1))
        graph[u].append((v, w))
        if not is_directed:
            graph[v].append((u, w))

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, weight in graph[current]:
            tentative_g = g_score[current] + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristiceuclidean(neighbor,end) if euclidean else heuristicmanhatten(neighbor,end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []
def reset_graph():
    global nodes, edges, weights, start_node, end_node, instruction, setting_nodes,traversal_type,adding_edges,edges_fixed,is_directed,weighted_graph,time_taken,first_node,waiting_for_heuristic,curr,euclidean,peak
    nodes = []
    edges = []
    weights = {}
    start_node = None
    end_node = None
    traversal_type=None
    instruction = "Press anywhere to add nodes, press Enter to proceed"
    setting_nodes = True
    adding_edges=False
    edges_fixed=False
    is_directed=None
    weighted_graph=None
    time_taken=None
    first_node=None
    waiting_for_heuristic=None
    euclidean=None
    curr=None
    peak=None
waiting_for_weight = False
input_text = ""

# Main Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            if RESET_BTN_RECT.collidepoint(mouse_pos):
                reset_graph()
                skip_next_click = True 
        if event.type == pygame.MOUSEBUTTONDOWN:
            if skip_next_click:
                    skip_next_click = False  # Ignore this click
                    continue
            x, y = pygame.mouse.get_pos()
            if event.button == 1 and not adding_edges and not edges_fixed:
                nodes.append((x, y))
            elif adding_edges and not edges_fixed and not waiting_for_weight:
                for i, (nx, ny) in enumerate(nodes):
                    if math.hypot(x - nx, y - ny) < 8:
                        if first_node is None:
                            first_node = i
                        elif i != first_node:
                            edges.append((first_node, i))
                            if weighted_graph:
                                waiting_for_weight = (first_node, i)
                                instruction = f"Enter weight for edge {first_node + 1}-{i + 1} and press Enter"
                            first_node = None
                        break
            elif event.button == 3 and edges_fixed:
                for i, (nx, ny) in enumerate(nodes):
                    if math.hypot(x - nx, y - ny) < 8:
                        if traversal_type in ['Dijkstra', 'A*']:
                            if start_node is None:
                                start_node = i
                                instruction = "Start node selected (green). Right-click to select end node"
                            elif end_node is None and i != start_node:
                                end_node = i
                                instruction = f"End node selected (red). Press Space to start {traversal_type} traversal"
                        else:
                            start_node = i
                            instruction = f"{traversal_type} root selected. Press Space for traversal"
                        break

        if event.type == pygame.KEYDOWN:
            if waiting_for_weight:
                if event.key == pygame.K_RETURN:
                    try:
                        weight = int(input_text)
                        weights[waiting_for_weight] = weight
                        waiting_for_weight = False
                        input_text = ""
                        instruction = "Join the edges, press Enter to proceed"
                    except ValueError:
                        instruction = "Invalid weight! Enter a number and press Enter"
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode
            else:
                if event.key == pygame.K_RETURN:
                    if adding_edges:
                        if weighted_graph and len(weights) < len(edges):
                            instruction = "Please assign weights to all edges before proceeding"
                        else:
                            edges_fixed = True
                            adding_edges = False
                            if weighted_graph:
                                instruction = "Press D for DFS, B for BFS, J for Dijkstra, or A for A*"
                            else:
                                instruction = "Press D for DFS or B for BFS"
                    elif not adding_edges and weighted_graph is None and is_directed is None:
                        instruction = "Press D for directed, U for undirected"   
                    elif not adding_edges and weighted_graph is None:
                        instruction = "Press W for weighted, U for unweighted"
                    else:
                        adding_edges = True
                        instruction = "Join the edges, press Enter to proceed"
                elif event.key == pygame.K_d and weighted_graph is None and is_directed is None:
                    is_directed = True
                    instruction = "Directed graph selected. Press Enter"
                elif event.key == pygame.K_u and weighted_graph is None and is_directed is None:
                    is_directed = False
                    instruction = "Undirected graph selected. Press Enter"
                elif event.key == pygame.K_w and weighted_graph is None:
                    weighted_graph = True
                    instruction = "Weighted graph selected. Press Enter to add edges"
                elif event.key == pygame.K_u and weighted_graph is None:
                    weighted_graph = False
                    instruction = "Unweighted graph selected. Press Enter to add edges"
                elif event.key == pygame.K_d and edges_fixed:
                    traversal_type = 'DFS'
                    instruction = "DFS selected. Right-click to select root node, press Space to start traversal"
                    start_node = None
                    end_node = None
                elif event.key == pygame.K_b and edges_fixed:
                    traversal_type = 'BFS'
                    instruction = "BFS selected. Right-click to select root node, press Space to start traversal"
                    start_node = None
                    end_node = None
                elif event.key == pygame.K_j and edges_fixed and weighted_graph:
                    traversal_type = 'Dijkstra'
                    instruction = "Dijkstra selected. Right-click to select START then END node"
                    start_node = None
                    end_node = None
                elif event.key == pygame.K_a and edges_fixed and weighted_graph:
                    traversal_type = 'A*'
                    instruction = "A* selected. Press 'E' for Euclidean or 'M' for Manhattan heuristic."
                    start_node = None
                    end_node = None
                    waiting_for_heuristic = True
                if waiting_for_heuristic:
                    if event.key == pygame.K_e:
                        euclidean = True
                        instruction = "Euclidean selected. Press Enter to select start and end node."
                    elif event.key == pygame.K_m:
                        euclidean = False
                        instruction = "Manhattan selected. Press Enter to select start and end node."
                    elif event.key == pygame.K_RETURN and euclidean is not None:
                        instruction = "Right-click to select START node"
                        waiting_for_heuristic = False
                        
                elif event.key == pygame.K_SPACE and start_node is not None:
                    if traversal_type in ['Dijkstra', 'A*'] and end_node is None:
                        instruction = "Please select END node by right-clicking"
                        continue
                    tracemalloc.start()
                    start_time = time.time()
                    if traversal_type == 'DFS':
                        path = dfs(start_node)
                    elif traversal_type == 'BFS':
                        path = bfs(start_node)
                    elif traversal_type == 'Dijkstra' and end_node is not None:
                        path = dijkstra(start_node, end_node)
                    elif traversal_type == 'A*' and end_node is not None:
                        path = a_star(start_node, end_node)
                    else:
                        path = []
                    end_time = time.time()
                    curr, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    time_taken = end_time - start_time
                    for p in path:
                        pygame.draw.circle(screen, YELLOW, nodes[p], 8)
                        pygame.display.flip()
                        pygame.time.delay(200)
                    start_node = None
                    end_node = None
                    traversal_type = None

                    if weighted_graph:
                        instruction = "Press D for DFS, B for BFS, J for Dijkstra, or A for A*"
                    else:
                        instruction = "Press D for DFS or B for BFS"
    draw_graph()