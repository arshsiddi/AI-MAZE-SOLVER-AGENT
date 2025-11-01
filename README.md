AI Maze Solver Agent
This project is an advanced pathfinding visualizer built with Python and Tkinter. It allows users to visualize a wide variety of search algorithms on a 2D grid, from standard searches like A\* and BFS to more advanced local search and evolutionary algorithms.

The core "AI" feature of this agent is its ability to train a machine learning model (a neural network) to learn its own pathfinding heuristic. You can then compare the performance of this learned heuristic against traditional ones like Manhattan or Euclidean distance.

📜 Features
Interactive Grid: Draw and erase obstacles by clicking and dragging.

Start & Goal: Set start (left-click) and goal (right-click) nodes.

Real-time Visualization: Watch algorithms explore the maze with adjustable animation speed.

Controls: Start, stop, and reset the grid at any time.

Maze Generation: Instantly generate a random maze.

Statistics: Reports path length, visited nodes, and execution time after each search.

🧠 The Learned Heuristic (MLP)
This project's most unique feature is the "Train Heuristic Model" button.

Training: Clicking it generates 50+ random mazes. For each maze, it runs an exhaustive BFS from the goal to find the true optimal path cost from every single reachable cell to that goal.

Feature Extraction: It creates a training dataset where the features are [node_x, node_y, delta_x_to_goal, delta_y_to_goal, euclidean_dist_to_goal].

Model: It uses this data to train a scikit-learn MLPRegressor (a Multi-Layer Perceptron, or simple neural network) to predict the optimal path cost given a node's features.

Persistence: The trained model is saved as learned_heuristic.joblib and automatically loaded the next time you start the app.

Usage: You can select "Learned (MLP)" from the heuristic dropdown to have A\* or Greedy Best-First Search use your trained AI model as its guide.

🧠 Implemented Algorithms
A comprehensive suite of 12 algorithms is included, grouped by strategy:

Standard Search (Informed & Uninformed)

A\* Search

Greedy Best-First Search

Breadth-First Search (BFS)

Depth-First Search (DFS)

Uniform-Cost Search (UCS)

Local Search

HC (Steepest Ascent)

HC (Simple)

HC (Stochastic)

HC (Random Restart)

Simulated Annealing

Evolutionary Algorithms

Genetic Algorithm

Planning Strategies

Forward Search (A\*)

Backward Search (A\*)

Hierarchical Planning (uses A\* to navigate to predefined waypoints)

🛠️ Technologies Used
Python 3

Tkinter: For the graphical user interface.

scikit-learn: For the MLPRegressor neural network.

NumPy: For high-performance numerical operations for the ML model.

joblib: For saving and loading the trained ML model.

🚀 Getting Started

1. Clone the Repository
   Bash

git clone https://github.com/your-username/ai-maze-solver.git
cd ai-maze-solver
(Remember to replace your-username with your actual GitHub username!)

2. Create a Virtual Environment (Recommended)
   Bash

# On macOS/Linux

python3 -m venv venv
source venv/bin/activate

# On Windows

python -m venv venv
.\venv\Scripts\activate 3. Install Required Libraries
This project requires scikit-learn, numpy, and joblib. You can install them using pip. Tkinter is usually included with Python, so no separate installation is needed.

Here is the installation command:

Bash

pip install scikit-learn numpy joblib 4. Run the Application
Save the code as maze_solver.py (or your preferred name) and run:

Bash

python maze_solver.py
📖 How to Use
Train the AI (Optional but Recommended)

Click the "Train Heuristic Model" button on the left panel.

Wait for the process to complete (it may take a minute or two). You will see progress in your console and a pop-up when it's done.

The "Model: N/A" label will change to "Model: Trained & Saved".

Set Up the Maze

Generate: Click "Generate Random Maze" for a quick setup.

Manual:

Left-Click: Set the Start node (green).

Right-Click: Set the Goal node (red).

Click and Drag: Draw Obstacles (dark blue).

Select & Run

Choose an Algorithm from the first dropdown.

If you chose a heuristic algorithm (like A\*), select a Heuristic from the second dropdown (e.g., "Learned (MLP)").

Adjust the Animation Speed slider (drag left for faster, right for slower).

Click "Start".

Observe

Watch the algorithm work. Visited nodes are gray, and the frontier (open set) is light blue.

The final path will be drawn in yellow.

The stats (path length, nodes visited, time) will update below.

Click "Stop" to interrupt a search or "Reset Grid" to clear everything.
