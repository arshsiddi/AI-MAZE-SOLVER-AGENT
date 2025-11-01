import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import collections
import random
import time
import math
import os # --- NEW ---

# --- NEW: Machine Learning Imports ---
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

# --- Constants ---
GRID_WIDTH = 50
GRID_HEIGHT = 40
CELL_SIZE = 15
ANIMATION_DELAY_MIN = 1
ANIMATION_DELAY_MAX = 200
MODEL_FILENAME = "learned_heuristic.joblib" # --- NEW ---


# --- Genetic Algorithm Constants ---
GA_POPULATION_SIZE = 100
GA_MAX_GENERATIONS = 50
GA_MUTATION_RATE = 0.1

# --- Colors ---
COLOR_BG = "#2c3e50"
COLOR_EMPTY = "#ecf0f1"
COLOR_OBSTACLE = "#34495e"
COLOR_START = "#27ae60"
COLOR_GOAL = "#c03b2b"
COLOR_PATH = "#f1c40f"
COLOR_FRONTIER = "#3498db"
COLOR_VISITED = "#95a5a6"
COLOR_TEXT = "#ecf0f1"
COLOR_FRAME_BG = "#34495e"
COLOR_BUTTON = "#2980b9"
COLOR_BUTTON_ACTIVE = "#3498db"
COLOR_STOP_BUTTON = "#e74c3c"
COLOR_STOP_BUTTON_ACTIVE = "#c0392b"


class MazeSolverGUI:
    """
    A GUI application for visualizing various pathfinding algorithms on a 2D grid maze.
    """

    def __init__(self, root):
        """Initialize the main application window and its components."""
        self.root = root
        self.root.title("AI Maze Solver Agent")
        self.root.configure(bg=COLOR_BG)

        # --- Maze State ---
        self.grid = []
        self.start_node = None
        self.goal_node = None
        self.searching = False
        self.force_stop = False

        # --- NEW: Learned Heuristic Model ---
        self.learned_model = None

        # --- UI Control Variables ---
        self.animation_delay = tk.IntVar(value=10)
        self.sa_temperature = tk.DoubleVar(value=1000.0)
        self.sa_cooling_rate = tk.DoubleVar(value=0.995)
        self.heuristic_weight = tk.DoubleVar(value=1.0)
        self.hc_restarts = tk.IntVar(value=5)

        # --- Algorithm & Heuristic Selection ---
        self.algorithms = {
            # Standard Search
            "A* Search": self.a_star,
            "Greedy Best-First Search": self.greedy_best_first,
            "Breadth-First Search (BFS)": self.bfs,
            "Depth-First Search (DFS)": self.dfs,
            "Uniform-Cost Search (UCS)": self.ucs,
            # Local Search
            "HC (Steepest Ascent)": self.hill_climbing_steepest_ascent,
            "HC (Simple)": self.hill_climbing_simple,
            "HC (Stochastic)": self.hill_climbing_stochastic,
            "HC (Random Restart)": self.hill_climbing_random_restart,
            "Simulated Annealing": self.simulated_annealing,
            # Evolutionary
            "Genetic Algorithm": self.genetic_algorithm,
            # Planning Strategies
            "Forward Search (A*)": self.a_star,
            "Backward Search (A*)": self.backward_search,
            "Hierarchical Planning": self.hierarchical_planning,
        }
        self.selected_algorithm = tk.StringVar(value="A* Search")
        self.selected_algorithm.trace("w", self._update_control_state)

        # --- MODIFIED: Added Learned Heuristic ---
        self.heuristics = {
            "Manhattan": self.heuristic_manhattan,
            "Euclidean": self.heuristic_euclidean,
            "Learned (MLP)": self.heuristic_learned,
        }
        self.selected_heuristic = tk.StringVar(value="Manhattan")


        # --- UI Setup ---
        self.setup_ui()
        self.reset_grid()
        self.load_learned_model() # --- NEW ---

    def setup_ui(self):
        """Creates and places all the UI widgets."""
        control_frame = tk.Frame(self.root, bg=COLOR_FRAME_BG, padx=10, pady=10)
        control_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        self.canvas = tk.Canvas(
            self.root,
            width=GRID_WIDTH * CELL_SIZE,
            height=GRID_HEIGHT * CELL_SIZE,
            bg=COLOR_OBSTACLE,
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=1, padx=10, pady=10)

        # --- Algorithm & Heuristic Controls ---
        tk.Label(control_frame, text="Algorithm:", bg=COLOR_FRAME_BG, fg=COLOR_TEXT).pack(pady=(0, 5))
        self.algo_menu = ttk.OptionMenu(control_frame, self.selected_algorithm, self.selected_algorithm.get(), *self.algorithms.keys())
        self.algo_menu.pack(fill='x', pady=5)

        tk.Label(control_frame, text="Heuristic:", bg=COLOR_FRAME_BG, fg=COLOR_TEXT).pack(pady=(10, 5))
        self.heuristic_menu = ttk.OptionMenu(control_frame, self.selected_heuristic, self.selected_heuristic.get(), *self.heuristics.keys())
        self.heuristic_menu.pack(fill='x', pady=5)

        # --- Main Action Buttons ---
        buttons_frame = tk.Frame(control_frame, bg=COLOR_FRAME_BG)
        buttons_frame.pack(fill='x', pady=5, ipady=4)
        self.start_button = tk.Button(buttons_frame, text="Start", command=self.start_search, bg=COLOR_BUTTON, fg=COLOR_TEXT, activebackground=COLOR_BUTTON_ACTIVE, relief=tk.FLAT)
        self.start_button.pack(side=tk.LEFT, fill='x', expand=True, padx=(0,2))
        self.stop_button = tk.Button(buttons_frame, text="Stop", command=self.stop_search, bg=COLOR_STOP_BUTTON, fg=COLOR_TEXT, activebackground=COLOR_STOP_BUTTON_ACTIVE, relief=tk.FLAT, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill='x', expand=True, padx=(2,0))

        self.reset_button = tk.Button(control_frame, text="Reset Grid", command=self.reset_grid, bg=COLOR_BUTTON, fg=COLOR_TEXT, activebackground=COLOR_BUTTON_ACTIVE, relief=tk.FLAT)
        self.reset_button.pack(fill='x', pady=5, ipady=4)
        self.generate_button = tk.Button(control_frame, text="Generate Random Maze", command=self.generate_random_maze, bg=COLOR_BUTTON, fg=COLOR_TEXT, activebackground=COLOR_BUTTON_ACTIVE, relief=tk.FLAT)
        self.generate_button.pack(fill='x', pady=5, ipady=4)

        # --- NEW: Train Model Button ---
        self.train_button = tk.Button(control_frame, text="Train Heuristic Model", command=self.train_heuristic_model_gui, bg="#16a085", fg=COLOR_TEXT, activebackground="#1abc9c", relief=tk.FLAT)
        self.train_button.pack(fill='x', pady=5, ipady=4)


        # --- Sliders ---
        tk.Label(control_frame, text="Animation Speed:", bg=COLOR_FRAME_BG, fg=COLOR_TEXT).pack(pady=(15, 5))
        self.speed_slider = tk.Scale(control_frame, from_=ANIMATION_DELAY_MIN, to=ANIMATION_DELAY_MAX, orient=tk.HORIZONTAL, variable=self.animation_delay, bg=COLOR_FRAME_BG, fg=COLOR_TEXT, highlightthickness=0)
        self.speed_slider.pack(fill='x', pady=5)

        tk.Label(control_frame, text="Heuristic Weight (>1 Overestimates)", bg=COLOR_FRAME_BG, fg=COLOR_TEXT).pack(pady=(15, 5))
        self.heuristic_slider = tk.Scale(control_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL, variable=self.heuristic_weight, bg=COLOR_FRAME_BG, fg=COLOR_TEXT, highlightthickness=0, resolution=0.1)
        self.heuristic_slider.pack(fill='x', pady=5)

        tk.Label(control_frame, text="SA Initial Temperature:", bg=COLOR_FRAME_BG, fg=COLOR_TEXT).pack(pady=(15, 5))
        self.sa_temp_slider = tk.Scale(control_frame, from_=100, to=10000, orient=tk.HORIZONTAL, variable=self.sa_temperature, bg=COLOR_FRAME_BG, fg=COLOR_TEXT, highlightthickness=0, resolution=100)
        self.sa_temp_slider.pack(fill='x', pady=5)

        tk.Label(control_frame, text="SA Cooling Rate:", bg=COLOR_FRAME_BG, fg=COLOR_TEXT).pack(pady=(15, 5))
        self.sa_cool_slider = tk.Scale(control_frame, from_=0.9, to=0.999, orient=tk.HORIZONTAL, variable=self.sa_cooling_rate, bg=COLOR_FRAME_BG, fg=COLOR_TEXT, highlightthickness=0, resolution=0.001)
        self.sa_cool_slider.pack(fill='x', pady=5)

        tk.Label(control_frame, text="HC Random Restarts:", bg=COLOR_FRAME_BG, fg=COLOR_TEXT).pack(pady=(15, 5))
        self.hc_restart_slider = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.hc_restarts, bg=COLOR_FRAME_BG, fg=COLOR_TEXT, highlightthickness=0)
        self.hc_restart_slider.pack(fill='x', pady=5)

        # --- Info & Stats ---
        # --- NEW: Model Status Label ---
        self.model_status_label = tk.Label(control_frame, text="Model: N/A", bg=COLOR_FRAME_BG, fg=COLOR_TEXT, justify=tk.LEFT)
        self.model_status_label.pack(pady=(15, 0), fill='x')
        self.stats_label = tk.Label(control_frame, text="Path Length: N/A\nVisited Nodes: N/A\nTime: N/A", bg=COLOR_FRAME_BG, fg=COLOR_TEXT, justify=tk.LEFT)
        self.stats_label.pack(pady=5, fill='x')

        self.canvas.bind("<Button-1>", self.handle_mouse_click)
        self.canvas.bind("<Button-3>", self.handle_mouse_click)
        self.canvas.bind("<B1-Motion>", self.handle_mouse_drag)
        self._update_control_state()

    # ... (rest of the UI and grid functions remain the same) ...
    def reset_grid(self):
        """Resets the grid to an empty state and redraws it."""
        if self.searching: return
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.start_node = (5, GRID_HEIGHT // 2)
        self.goal_node = (GRID_WIDTH - 6, GRID_HEIGHT // 2)
        self.stats_label.config(text="Path Length: N/A\nVisited Nodes: N/A\nTime: N/A")
        self.draw_grid()
    # ...
    # --- HEURISTICS SECTION ---
    # --- MODIFIED: Added Learned Heuristic ---

    def heuristic(self, a, b):
        """Dispatcher for the selected heuristic, applying a weight."""
        heuristic_func = self.heuristics[self.selected_heuristic.get()]
        return heuristic_func(a, b) * self.heuristic_weight.get()

    def heuristic_manhattan(self, a, b):
        (x1, y1), (x2, y2) = a, b
        return abs(x1 - x2) + abs(y1 - y2)

    def heuristic_euclidean(self, a, b):
        (x1, y1), (x2, y2) = a, b
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def heuristic_learned(self, a, b):
        """
        Uses the trained MLP model to predict the heuristic cost.
        'a' is the current node, 'b' is the goal node.
        """
        if self.learned_model is None:
            # Fallback to Manhattan if model isn't loaded/trained
            return self.heuristic_manhattan(a, b)

        # Extract features for node 'a' relative to the goal 'b'
        features = self._extract_features(a, b)

        # scikit-learn expects a 2D array for prediction
        features_2d = np.array(features).reshape(1, -1)

        prediction = self.learned_model.predict(features_2d)
        return prediction[0]


    # --- NEW: HEURISTIC LEARNING METHODS ---

    def _extract_features(self, node, goal):
        """Extracts a feature vector for a given node and goal."""
        x, y = node
        gx, gy = goal

        # Feature set: coordinates, and manhattan/euclidean distances to goal
        return [
            x, y,
            abs(x - gx),
            abs(y - gy),
            math.sqrt((x - gx)**2 + (y - gy)**2)
        ]

    def _generate_training_data_for_maze(self, temp_grid, temp_goal):
        """Generates training samples for a single maze configuration."""
        # Use BFS starting from the GOAL to find the true shortest path cost
        q = collections.deque([(temp_goal, 0)])
        visited = {temp_goal: 0}

        while q:
            (x, y), cost = q.popleft()

            # Check neighbors
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx, ny) not in visited and temp_grid[ny][nx] == 0:
                    visited[(nx, ny)] = cost + 1
                    q.append(((nx, ny), cost + 1))

        # Create training samples from the calculated costs
        X_maze, y_maze = [], []
        for node, true_cost in visited.items():
            X_maze.append(self._extract_features(node, temp_goal))
            y_maze.append(true_cost)

        return X_maze, y_maze

    def train_heuristic_model_gui(self):
        """GUI wrapper for training, provides user feedback."""
        if self.searching:
            messagebox.showwarning("Busy", "Cannot train while a search is running.")
            return

        self.model_status_label.config(text="Model: Training...")
        self.root.update()

        try:
            num_mazes = 50 # Increase for a better model
            X_train, y_train = [], []

            print(f"Generating training data from {num_mazes} mazes...")
            for i in range(num_mazes):
                # Create a temporary random maze
                temp_grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
                temp_goal = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

                for y in range(GRID_HEIGHT):
                    for x in range(GRID_WIDTH):
                        if (x, y) != temp_goal and random.random() < 0.3:
                            temp_grid[y][x] = 1

                # Make sure goal is not an obstacle
                temp_grid[temp_goal[1]][temp_goal[0]] = 0

                X_maze, y_maze = self._generate_training_data_for_maze(temp_grid, temp_goal)
                X_train.extend(X_maze)
                y_train.extend(y_maze)

            print(f"Training model on {len(y_train)} samples...")

            # Define and train the Neural Network
            # A simple MLP: 2 hidden layers with 32 and 16 neurons
            self.learned_model = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                verbose=True # Prints progress to console
            )
            self.learned_model.fit(np.array(X_train), np.array(y_train))

            # Save the trained model to a file
            joblib.dump(self.learned_model, MODEL_FILENAME)

            self.model_status_label.config(text=f"Model: Trained & Saved")
            print("Training complete and model saved.")
            messagebox.showinfo("Success", f"Heuristic model trained on {len(y_train)} samples and saved as {MODEL_FILENAME}")

        except Exception as e:
            self.model_status_label.config(text="Model: Training Failed")
            messagebox.showerror("Error", f"An error occurred during training: {e}")
            print(f"Training failed: {e}")

    def load_learned_model(self):
        """Loads the pre-trained heuristic model from a file if it exists."""
        if os.path.exists(MODEL_FILENAME):
            try:
                self.learned_model = joblib.load(MODEL_FILENAME)
                self.model_status_label.config(text="Model: Loaded from file")
                print(f"Loaded heuristic model from {MODEL_FILENAME}")
            except Exception as e:
                self.learned_model = None
                self.model_status_label.config(text="Model: Load failed")
                print(f"Error loading model: {e}")
        else:
            self.learned_model = None
            self.model_status_label.config(text="Model: Not trained")
            print("No pre-trained model found.")

    # All other methods (bfs, dfs, a_star, etc.) remain exactly the same as before.
    # ... (paste all your existing algorithm implementations here)
    def _update_control_state(self, *args):
        """Enable/disable sliders based on the selected algorithm."""
        algo = self.selected_algorithm.get()
        is_hc_restart = algo == "HC (Random Restart)"
        is_sa = algo == "Simulated Annealing"

        self.hc_restart_slider.config(state=tk.NORMAL if is_hc_restart else tk.DISABLED)
        self.sa_temp_slider.config(state=tk.NORMAL if is_sa else tk.DISABLED)
        self.sa_cool_slider.config(state=tk.NORMAL if is_sa else tk.DISABLED)
    def draw_grid(self, path=None, frontier=None, visited=None):
        """Draws the entire maze grid on the canvas."""
        self.canvas.delete("all")
        frontier = frontier or set()
        visited = visited or set()
        path = path or []
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pos = (x, y)
                x1, y1, x2, y2 = x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE
                color = COLOR_EMPTY
                if self.grid[y][x] == 1: color = COLOR_OBSTACLE
                elif pos == self.start_node: color = COLOR_START
                elif pos == self.goal_node: color = COLOR_GOAL
                elif pos in path: color = COLOR_PATH
                elif pos in frontier: color = COLOR_FRONTIER
                elif pos in visited: color = COLOR_VISITED
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=COLOR_OBSTACLE)

    def handle_mouse_click(self, event):
        """Handles setting start/goal nodes with mouse clicks."""
        if self.searching: return
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            if event.num == 1 and (x,y) != self.goal_node: self.start_node = (x, y)
            elif event.num == 3 and (x,y) != self.start_node: self.goal_node = (x, y)
            self.draw_grid()

    def handle_mouse_drag(self, event):
        """Handles drawing obstacles by dragging the mouse."""
        if self.searching: return
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            if (x, y) != self.start_node and (x, y) != self.goal_node:
                self.grid[y][x] = 1
                self.draw_grid()

    def generate_random_maze(self):
        """Generates a simple random maze by placing obstacles."""
        if self.searching: return
        self.reset_grid()
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if (x, y) != self.start_node and (x, y) != self.goal_node:
                    if random.random() < 0.3: self.grid[y][x] = 1
        self.draw_grid()

    def start_search(self):
        """Initiates the selected search algorithm."""
        if self.searching: return
        if not self.start_node or not self.goal_node:
            messagebox.showerror("Error", "Please set both start and goal nodes.")
            return

        # --- NEW: Check for learned model selection ---
        if self.selected_heuristic.get() == "Learned (MLP)" and self.learned_model is None:
            messagebox.showwarning("Model Not Found", "The learned heuristic is selected, but no model has been trained or loaded. Please train a model first. Falling back to Manhattan distance.")

        self.searching = True
        self.force_stop = False
        self.toggle_controls(False)
        self.stats_label.config(text="Path Length: Searching...\nVisited Nodes: 0\nTime: ...")

        self.start_time = time.time()
        algorithm_func = self.algorithms[self.selected_algorithm.get()]
        search_generator = algorithm_func()
        self.run_animation(search_generator)

    def stop_search(self):
        """Forces the current search to stop."""
        if self.searching:
            self.force_stop = True

    def run_animation(self, generator):
        """Steps through the search generator to animate the process."""
        if self.force_stop:
            self.searching = False
            self.toggle_controls(True)
            messagebox.showinfo("Stopped", "Search was interrupted by the user.")
            return
        try:
            yield_data = next(generator)
            algo_name = self.selected_algorithm.get()
            if algo_name == "Genetic Algorithm":
                best_path, all_nodes_in_pop = yield_data
                self.draw_grid(path=best_path, visited=all_nodes_in_pop)
                self.stats_label.config(text=f"Best Path: {len(best_path)}\nPopulation Coverage: {len(all_nodes_in_pop)}\nTime: ...")
            elif algo_name == "Hierarchical Planning":
                path_so_far, visited = yield_data
                self.draw_grid(path=path_so_far, visited=visited)
                self.stats_label.config(text=f"Path Length: Searching...\nVisited Nodes: {len(visited)}\nTime: ...")
            else:
                frontier, visited = yield_data
                self.draw_grid(frontier=frontier, visited=visited)
                self.stats_label.config(text=f"Path Length: Searching...\nVisited Nodes: {len(visited)}\nTime: ...")

            self.root.after(self.animation_delay.get(), lambda: self.run_animation(generator))
        except StopIteration as e:
            end_time = time.time()
            duration = end_time - self.start_time
            path = e.value
            if path:
                self.draw_grid(path=path)
                self.stats_label.config(text=f"Path Length: {len(path)}\nVisited Nodes: {getattr(self, 'final_visited_count', 0)}\nTime: {duration:.4f}s")
            else:
                messagebox.showinfo("Result", "No path found.")
                self.stats_label.config(text=f"Path Length: N/A\nVisited Nodes: {getattr(self, 'final_visited_count', 0)}\nTime: {duration:.4f}s")
            self.searching = False
            self.toggle_controls(True)

    def toggle_controls(self, enabled):
        """Enables or disables UI controls during search."""
        standard_state = tk.NORMAL if enabled else tk.DISABLED
        stop_state = tk.NORMAL if not enabled else tk.DISABLED

        self.start_button.config(state=standard_state)
        self.stop_button.config(state=stop_state)
        self.reset_button.config(state=standard_state)
        self.generate_button.config(state=standard_state)
        self.algo_menu.config(state=standard_state)
        self.heuristic_menu.config(state=standard_state)
        self.speed_slider.config(state=standard_state)
        self.heuristic_slider.config(state=standard_state)
        self.train_button.config(state=standard_state) # --- MODIFIED ---

        if enabled:
            self._update_control_state()
        else:
            self.hc_restart_slider.config(state=tk.DISABLED)
            self.sa_temp_slider.config(state=tk.DISABLED)
            self.sa_cool_slider.config(state=tk.DISABLED)

    def get_neighbors(self, node):
        """Gets valid neighbors for a given node."""
        x, y = node
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.grid[ny][nx] == 0]

    def reconstruct_path(self, came_from, current):
        """Reconstructs the path from the goal back to the start."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        if current is not None:
              path.append(current)
        return path[::-1]

    # ... Paste all your other algorithm implementations here ...
    # bfs, dfs, ucs, a_star, hill_climbing, etc.
    def bfs(self):
        frontier = collections.deque([self.start_node])
        came_from = {self.start_node: None}
        visited = {self.start_node}
        while frontier:
            yield set(frontier), visited
            current = frontier.popleft()
            if current == self.goal_node:
                self.final_visited_count = len(visited)
                return self.reconstruct_path(came_from, current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    frontier.append(neighbor)
                    visited.add(neighbor)
        self.final_visited_count = len(visited)
        return None

    def dfs(self):
        frontier = [self.start_node]
        came_from = {self.start_node: None}
        visited = {self.start_node}
        while frontier:
            yield set(frontier), visited
            current = frontier.pop()
            if current == self.goal_node:
                self.final_visited_count = len(visited)
                return self.reconstruct_path(came_from, current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    frontier.append(neighbor)
                    visited.add(neighbor)
        self.final_visited_count = len(visited)
        return None

    def ucs(self):
        frontier = [(0, self.start_node)]
        heapq.heapify(frontier)
        came_from = {self.start_node: None}
        cost_so_far = {self.start_node: 0}
        frontier_set, visited_set = {self.start_node}, set()
        while frontier:
            yield frontier_set, visited_set
            _, current_node = heapq.heappop(frontier)
            if current_node in frontier_set: frontier_set.remove(current_node)
            visited_set.add(current_node)
            if current_node == self.goal_node:
                self.final_visited_count = len(visited_set)
                return self.reconstruct_path(came_from, current_node)
            for neighbor in self.get_neighbors(current_node):
                new_cost = cost_so_far[current_node] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(frontier, (priority, neighbor))
                    frontier_set.add(neighbor)
                    came_from[neighbor] = current_node
        self.final_visited_count = len(visited_set)
        return None

    def greedy_best_first(self):
        frontier = [(0, self.start_node)]
        heapq.heapify(frontier)
        came_from = {self.start_node: None}
        visited = {self.start_node}
        frontier_set = {self.start_node}
        while frontier:
            yield frontier_set, visited
            _, current = heapq.heappop(frontier)
            if current in frontier_set: frontier_set.remove(current)
            if current == self.goal_node:
                self.final_visited_count = len(visited)
                return self.reconstruct_path(came_from, current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    priority = self.heuristic(neighbor, self.goal_node) # Corrected order
                    heapq.heappush(frontier, (priority, neighbor))
                    frontier_set.add(neighbor)
                    visited.add(neighbor)
                    came_from[neighbor] = current
        self.final_visited_count = len(visited)
        return None

    def a_star(self):
        gen = self._a_star_segment(self.start_node, self.goal_node)
        try:
            while True:
                val = next(gen)
                yield val
        except StopIteration as e:
            return e.value

    def hill_climbing_steepest_ascent(self):
        current = self.start_node
        path = [current]
        visited_nodes_hc = {current}
        while current != self.goal_node:
            yield set(), set(path)
            unvisited_neighbors = [n for n in self.get_neighbors(current) if n not in visited_nodes_hc]
            if not unvisited_neighbors:
                self.final_visited_count = len(path)
                return None
            best_neighbor = min(unvisited_neighbors, key=lambda n: self.heuristic(n, self.goal_node))
            if self.heuristic(best_neighbor, self.goal_node) >= self.heuristic(current, self.goal_node):
                self.final_visited_count = len(path)
                return None
            current = best_neighbor
            path.append(current)
            visited_nodes_hc.add(current)
        self.final_visited_count = len(path)
        return path

    def hill_climbing_simple(self):
        current = self.start_node
        path = [current]
        visited_nodes_hc = {current}
        while current != self.goal_node:
            yield set(), set(path)
            moved = False
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited_nodes_hc and self.heuristic(neighbor, self.goal_node) < self.heuristic(current, self.goal_node):
                    current = neighbor
                    path.append(current)
                    visited_nodes_hc.add(current)
                    moved = True
                    break
            if not moved:
                self.final_visited_count = len(path)
                return None
        self.final_visited_count = len(path)
        return path

    def hill_climbing_stochastic(self):
        current = self.start_node
        path = [current]
        visited_nodes_hc = {current}
        while current != self.goal_node:
            yield set(), set(path)
            better_neighbors = [
                n for n in self.get_neighbors(current)
                if n not in visited_nodes_hc and self.heuristic(n, self.goal_node) < self.heuristic(current, self.goal_node)
            ]
            if not better_neighbors:
                self.final_visited_count = len(path)
                return None
            current = random.choice(better_neighbors)
            path.append(current)
            visited_nodes_hc.add(current)
        self.final_visited_count = len(path)
        return path

    def hill_climbing_random_restart(self):
        best_path = None
        total_visited = set()
        for i in range(self.hc_restarts.get()):
            current = self.start_node if i == 0 else self.get_random_node()
            path = [current]
            visited_this_run = {current}
            while current != self.goal_node:
                total_visited.update(path)
                yield set(), total_visited
                unvisited_neighbors = [n for n in self.get_neighbors(current) if n not in visited_this_run]
                if not unvisited_neighbors:
                    break
                best_neighbor = min(unvisited_neighbors, key=lambda n: self.heuristic(n, self.goal_node))
                if self.heuristic(best_neighbor, self.goal_node) >= self.heuristic(current, self.goal_node):
                    break
                current = best_neighbor
                path.append(current)
                visited_this_run.add(current)
            if path and path[-1] == self.goal_node:
                self.final_visited_count = len(total_visited)
                # Found a solution, no need to continue restarting
                return path
        self.final_visited_count = len(total_visited)
        return None

    def get_random_node(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if self.grid[y][x] == 0:
                return (x, y)

    def simulated_annealing(self):
        current = self.start_node
        path = [current]
        temp = self.sa_temperature.get()
        cooling_rate = self.sa_cooling_rate.get()
        while temp > 1 and current != self.goal_node:
            yield set(), set(path)
            neighbors = self.get_neighbors(current)
            if not neighbors:
                self.final_visited_count = len(path)
                return None
            next_node = random.choice(neighbors)
            current_energy = self.heuristic(current, self.goal_node)
            next_energy = self.heuristic(next_node, self.goal_node)
            if next_energy < current_energy or random.random() < math.exp((current_energy - next_energy) / temp):
                current = next_node
                if current in path:
                    path = path[:path.index(current)+1]
                else:
                    path.append(current)
            temp *= cooling_rate
        self.final_visited_count = len(set(path))
        return path if current == self.goal_node else None

    def genetic_algorithm(self):
        def create_individual():
            path = [self.start_node]
            current = self.start_node
            max_path_len = GRID_WIDTH * GRID_HEIGHT
            for _ in range(max_path_len):
                if current == self.goal_node: break
                neighbors = self.get_neighbors(current)
                if not neighbors: break
                current = random.choice(neighbors)
                path.append(current)
            return path
        def calculate_fitness(path):
            if not path: return 0
            dist_to_goal = self.heuristic(path[-1], self.goal_node)
            length_penalty = len(path)
            is_valid = all(path[i] in self.get_neighbors(path[i-1]) for i in range(1, len(path)))
            if not is_valid: return 0.001
            fitness = 1000.0 / (dist_to_goal + length_penalty + 1)
            if path[-1] == self.goal_node:
                fitness *= 2
            return fitness
        def crossover(parent1, parent2):
            if not parent1 or not parent2 or min(len(parent1), len(parent2)) <= 1: return parent1, parent2
            point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        def mutate(path):
            if len(path) < 2: return path
            idx = random.randint(1, len(path)-1)
            neighbors = self.get_neighbors(path[idx-1])
            if neighbors:
                path[idx] = random.choice(neighbors)
                new_path = path[:idx+1]
                current = new_path[-1]
                for _ in range(GRID_WIDTH * 2):
                    if current == self.goal_node: break
                    neighbors = self.get_neighbors(current)
                    if not neighbors: break
                    current = random.choice(neighbors)
                    new_path.append(current)
                return new_path
            return path
        population = [create_individual() for _ in range(GA_POPULATION_SIZE)]
        best_path = []
        for gen in range(GA_MAX_GENERATIONS):
            fitness_scores = sorted([(ind, calculate_fitness(ind)) for ind in population], key=lambda x: x[1], reverse=True)
            best_path_current_gen, best_fitness = fitness_scores[0]
            if not best_path or calculate_fitness(best_path_current_gen) > calculate_fitness(best_path):
                best_path = best_path_current_gen
            all_nodes_in_pop = set(node for path in population for node in path)
            yield best_path, all_nodes_in_pop
            if best_path and best_path[-1] == self.goal_node:
                self.final_visited_count = len(all_nodes_in_pop)
                return best_path
            next_gen = [fs[0] for fs in fitness_scores[:int(0.1 * GA_POPULATION_SIZE)]]
            while len(next_gen) < GA_POPULATION_SIZE:
                parents = random.choices(fitness_scores, k=2, weights=[f[1] for f in fitness_scores])
                child1, child2 = crossover(parents[0][0], parents[1][0])
                if random.random() < GA_MUTATION_RATE: child1 = mutate(child1)
                if random.random() < GA_MUTATION_RATE: child2 = mutate(child2)
                next_gen.extend([child1, child2])
            population = next_gen[:GA_POPULATION_SIZE]
        self.final_visited_count = len(set(node for path in population for node in path))
        return best_path if best_path and best_path[-1] == self.goal_node else None

    def _a_star_segment(self, start_pos, goal_pos):
        frontier = [(0, start_pos)]
        heapq.heapify(frontier)
        came_from, cost_so_far = {start_pos: None}, {start_pos: 0}
        frontier_set, visited_set = {start_pos}, set()
        while frontier:
            yield frontier_set, visited_set
            _, current_node = heapq.heappop(frontier)
            if current_node in frontier_set: frontier_set.remove(current_node)
            visited_set.add(current_node)
            if current_node == goal_pos:
                self.final_visited_count = len(visited_set)
                return self.reconstruct_path(came_from, current_node)
            for neighbor in self.get_neighbors(current_node):
                new_cost = cost_so_far[current_node] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_pos)
                    heapq.heappush(frontier, (priority, neighbor))
                    frontier_set.add(neighbor)
                    came_from[neighbor] = current_node
        self.final_visited_count = len(visited_set)
        return None

    def backward_search(self):
        start_pos, goal_pos = self.goal_node, self.start_node
        frontier = [(0, start_pos)]
        heapq.heapify(frontier)
        came_from, cost_so_far = {start_pos: None}, {start_pos: 0}
        frontier_set, visited_set = {start_pos}, set()
        while frontier:
            yield frontier_set, visited_set
            _, current_node = heapq.heappop(frontier)
            if current_node in frontier_set: frontier_set.remove(current_node)
            visited_set.add(current_node)
            if current_node == goal_pos:
                self.final_visited_count = len(visited_set)
                path = self.reconstruct_path(came_from, current_node)
                return path[::-1]
            for neighbor in self.get_neighbors(current_node):
                new_cost = cost_so_far[current_node] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_pos)
                    heapq.heappush(frontier, (priority, neighbor))
                    frontier_set.add(neighbor)
                    came_from[neighbor] = current_node
        self.final_visited_count = len(visited_set)
        return None

    def hierarchical_planning(self):
        waypoints = [
            self.start_node,
            (GRID_WIDTH // 3, GRID_HEIGHT // 2),
            (GRID_WIDTH * 2 // 3, GRID_HEIGHT // 2),
            self.goal_node
        ]
        plan = []
        for wp in waypoints:
            if self.grid[wp[1]][wp[0]] == 1:
                plan.append(self._find_nearest_clear_cell(wp))
            else:
                plan.append(wp)
        full_path = []
        total_visited_for_stats = set()
        for i in range(len(plan) - 1):
            sub_start = plan[i]
            sub_goal = plan[i+1]
            segment_generator = self._a_star_segment(sub_start, sub_goal)
            sub_path = None
            try:
                while True:
                    frontier, visited = next(segment_generator)
                    total_visited_for_stats.update(visited)
                    yield full_path, visited
            except StopIteration as e:
                sub_path = e.value
            if not sub_path:
                self.final_visited_count = len(total_visited_for_stats)
                return None
            full_path.extend(sub_path if i == 0 else sub_path[1:])
        self.final_visited_count = len(total_visited_for_stats)
        return full_path

    def _find_nearest_clear_cell(self, node):
        if self.grid[node[1]][node[0]] == 0:
            return node
        q = collections.deque([node])
        visited = {node}
        while q:
            x, y = q.popleft()
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx,ny) not in visited:
                    if self.grid[ny][nx] == 0:
                        return (nx, ny)
                    q.append((nx,ny))
                    visited.add((nx,ny))
        return node

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeSolverGUI(root)
    root.mainloop()