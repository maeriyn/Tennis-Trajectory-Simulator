import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
import threading

class TennisServeSimulator:
    def __init__(self):
        # Court dimensions in meters
        self.court_length = 23.77  # Full court length
        self.court_width = 8.23  # Singles court width
        self.net_height = 0.914  # Height of the net
        self.service_line = 6.40  # Distance from net to service line
        
        # Ball properties
        self.ball_radius = 0.067  # meters
        self.ball_mass = 0.057  # kg
        self.gravity = 9.81  # m/s^2
        self.drag_coefficient = 0.47  # More accurate drag coefficient for tennis balls
        self.air_density = 1.225  # kg/m^3
        
        # Unit conversion factors
        self.KMH_TO_MS = 1/3.6  # km/h to m/s
        self.MS_TO_MPH = 2.237  # m/s to mph
        
        # Standard player reach (m)
        self.lateral_reach = 2.8  # Average player lateral reach in meters
        
        # Service directions (angles in degrees from center, adjusted for cross-court)
        self.directions = {
            "T": {"Deuce": -9, "Ad": 9},      # T serve angles for each side
            "Body": {"Deuce": -12, "Ad": 12},   # Body serve angles
            "Wide": {"Deuce": -18, "Ad": 18}    # Wide serve angles
        }

        # Add racquet and arm specifications
        self.racquet_length = 0.685  # Standard tennis racquet length in meters
        self.arm_reach_factor = 1.2  # Arm reach as multiplier of height

        # Default view angles for isometric view
        self.default_elev = 20
        self.default_azim = -60

    def calculate_release_point(self, height, serving_side):
        """Calculate serve release point based on player height, arm reach, and racquet length"""
        arm_reach = height * self.arm_reach_factor  # Arm reach is proportional to height
        # Release height includes player height, arm reach, and racquet length
        release_height = height + arm_reach + self.racquet_length
        # X position depends on serving side
        x_position = self.court_width/4 if serving_side == "Deuce" else -self.court_width/4
        return [x_position, 0, release_height]  # [x, y, z] coordinates
    
    def calculate_initial_velocity(self, speed, direction, serving_side):
        """Calculate initial velocity components based on speed and direction with optimized angle"""
        # Convert direction angle to radians - use serving side specific angle
        angle_horizontal_rad = np.radians(self.directions[direction][serving_side])
        
        # Calculate optimal vertical angle based on speed - steeper downward angles
        if speed > 40:  # Very fast serves (>144 km/h)
            angle_vertical_rad = np.radians(-8)  # Steeper angle for fast serves
        elif speed > 30:  # Fast serves (>108 km/h)
            angle_vertical_rad = np.radians(-7)
        elif speed > 20:  # Medium serves (>72 km/h)
            angle_vertical_rad = np.radians(-6)
        else:  # Slower serves
            angle_vertical_rad = np.radians(-5)
        
        # Calculate velocity components
        vx = speed * np.cos(angle_vertical_rad) * np.sin(angle_horizontal_rad)
        vy = speed * np.cos(angle_vertical_rad) * np.cos(angle_horizontal_rad)
        vz = speed * np.sin(angle_vertical_rad)
        
        return [vx, vy, vz]

    def simulate_trajectory(self, release_point, initial_velocity, dt=0.01, max_time=2.0):
        """Simulate the 3D trajectory of the serve with air resistance"""
        positions = [release_point]
        velocities = [initial_velocity]
        times = [0]
        
        current_position = release_point.copy()
        current_velocity = initial_velocity.copy()
        current_time = 0
        
        # Cross-sectional area of the ball for drag calculation
        area = np.pi * self.ball_radius**2
        drag_factor = 0.5 * self.drag_coefficient * self.air_density * area / self.ball_mass
        
        # Continue until the ball hits the ground or exceeds max time
        while current_position[2] > 0 and current_time < max_time:
            current_time += dt
            
            # Calculate drag force
            speed = np.linalg.norm(current_velocity)
            if speed > 0:
                drag_acceleration = [
                    -drag_factor * speed * current_velocity[0],
                    -drag_factor * speed * current_velocity[1],
                    -drag_factor * speed * current_velocity[2]
                ]
            else:
                drag_acceleration = [0, 0, 0]
            
            # Update velocity with gravity and drag
            current_velocity[0] += drag_acceleration[0] * dt
            current_velocity[1] += drag_acceleration[1] * dt
            current_velocity[2] += (drag_acceleration[2] - self.gravity) * dt
            
            # Update position
            current_position[0] += current_velocity[0] * dt
            current_position[1] += current_velocity[1] * dt
            current_position[2] += current_velocity[2] * dt
            
            # Check if the ball crosses the net
            if len(positions) > 1:
                prev_y = positions[-1][1]
                if prev_y < self.court_length/2 and current_position[1] >= self.court_length/2:
                    # Interpolate to find height at net
                    t_fraction = (self.court_length/2 - prev_y) / (current_position[1] - prev_y)
                    height_at_net = positions[-1][2] + t_fraction * (current_position[2] - positions[-1][2])
                    
                    # If the ball hits the net
                    if height_at_net < self.net_height:
                        # Make the ball stop at the net
                        net_position = [
                            positions[-1][0] + t_fraction * (current_position[0] - positions[-1][0]),
                            self.court_length/2,
                            height_at_net
                        ]
                        positions.append(net_position)
                        velocities.append([0, 0, 0])
                        times.append(current_time)
                        break
            
            positions.append(current_position.copy())
            velocities.append(current_velocity.copy())
            times.append(current_time)
        
        return positions, velocities, times

    def is_serve_in(self, trajectory):
        """Check if the serve lands in the service box"""
        # Find the point where the ball hits the ground
        landing_point = None
        
        for i in range(1, len(trajectory)):
            if trajectory[i-1][2] > 0 and trajectory[i][2] <= 0:
                # Interpolate to find exact landing point
                t = -trajectory[i-1][2] / (trajectory[i][2] - trajectory[i-1][2])
                landing_x = trajectory[i-1][0] + t * (trajectory[i][0] - trajectory[i-1][0])
                landing_y = trajectory[i-1][1] + t * (trajectory[i][1] - trajectory[i-1][1])
                landing_point = [landing_x, landing_y, 0]
                break
        
        if landing_point is None:
            return False, None
        
        # Check if the landing point is in the service box
        # Service box is from the net to service line, and half the court width
        half_width = self.court_width / 2
        is_in = (self.court_length/2 < landing_point[1] <= self.court_length/2 + self.service_line and 
                 -half_width <= landing_point[0] <= half_width)
        
        return is_in, landing_point

    def can_opponent_return(self, trajectory, times, opponent, landing_point):
        """Determine if the opponent can return the serve based on reaction time and reach"""
        if landing_point is None:
            return False, "The serve did not land properly."
        
        # Get opponent properties
        lateral_reach = self.lateral_reach
        
        # Calculate opponent position (centered at baseline with some distance)
        opponent_position = [0, self.court_length, 0]
        
        # Find time when ball crosses the baseline
        ball_at_baseline_idx = None
        for i in range(1, len(trajectory)):
            if trajectory[i-1][1] < self.court_length and trajectory[i][1] >= self.court_length:
                ball_at_baseline_idx = i
                break
        
        # If the ball never reaches the baseline
        if (ball_at_baseline_idx is None):
            return False, "The serve did not reach the baseline."
        
        # Interpolate to find ball position and time at baseline
        t = (self.court_length - trajectory[ball_at_baseline_idx-1][1]) / (trajectory[ball_at_baseline_idx][1] - trajectory[ball_at_baseline_idx-1][1])
        ball_baseline_x = trajectory[ball_at_baseline_idx-1][0] + t * (trajectory[ball_at_baseline_idx][0] - trajectory[ball_at_baseline_idx-1][0])
        ball_baseline_z = trajectory[ball_at_baseline_idx-1][2] + t * (trajectory[ball_at_baseline_idx][2] - trajectory[ball_at_baseline_idx-1][2])
        time_at_baseline = times[ball_at_baseline_idx-1] + t * (times[ball_at_baseline_idx] - times[ball_at_baseline_idx-1])
        
        # Calculate landing time
        landing_idx = None
        for i in range(1, len(trajectory)):
            if trajectory[i-1][2] > 0 and trajectory[i][2] <= 0:
                landing_idx = i
                break
        
        if landing_idx is None:
            return False, "Could not determine when the ball landed."
        
        # Interpolate to find exact landing time
        t = -trajectory[landing_idx-1][2] / (trajectory[landing_idx][2] - trajectory[landing_idx-1][2])
        landing_time = times[landing_idx-1] + t * (times[landing_idx] - times[landing_idx-1])
        
        # Calculate time from landing to baseline crossing
        time_between_landing_and_baseline = max(0, time_at_baseline - landing_time)
        
        # Calculate total time available to react
        total_time_available = landing_time
        if ball_baseline_z <= 0:  # If the ball is already on the ground at baseline
            total_time_available = landing_time
        else:
            total_time_available = time_at_baseline
        
        # Check if opponent has enough time to react
        time_to_reach = max(0, total_time_available)
        
        # Calculate how far the opponent can move in the available time
        # Assume a max lateral speed of 4 m/s (top players can move quite fast)
        max_lateral_movement = time_to_reach * 4.0
        
        # Calculate distance needed to move
        lateral_distance_needed = abs(ball_baseline_x - opponent_position[0])
        
        # Check if opponent can reach the ball
        can_reach = lateral_distance_needed <= (lateral_reach + max_lateral_movement)
        
        # Generate explanation
        if can_reach:
            explanation = (f"{opponent} can return the serve! "
                          f"Time to react: {total_time_available:.2f}s, "
                          f"Needs to cover: {lateral_distance_needed:.2f}m, "
                          f"Can cover: {lateral_reach + max_lateral_movement:.2f}m")
        else:
            explanation = (f"{opponent} cannot return the serve. "
                          f"Time to react: {total_time_available:.2f}s, "
                          f"Needs to cover: {lateral_distance_needed:.2f}m, "
                          f"Can only cover: {lateral_reach + max_lateral_movement:.2f}m")
        
        return can_reach, explanation

    def create_court_visualization(self):
        """Create a 3D visualization of the tennis court"""
        fig = plt.figure(figsize=(15, 10))  # Increased figure size
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the court
        self._plot_court(ax)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Tennis Serve Trajectory')
        
        # Set aspect ratio and limits
        ax.set_box_aspect([1, 2.5, 0.8])
        ax.set_xlim([-self.court_width/2 - 1, self.court_width/2 + 1])
        ax.set_ylim([0, self.court_length + 1])
        ax.set_zlim([0, 3])

        # Set initial view to isometric
        ax.view_init(elev=self.default_elev, azim=self.default_azim)
        
        return fig, ax

    def update_visualization(self, ax, trajectory, landing_point, opponent, opponent_return_result, opponent_explanation):
        """Update the visualization with trajectory data"""
        ax.clear()
        
        # Plot the court
        self._plot_court(ax)
        
        # Extract x, y, z coordinates for plotting
        x = [point[0] for point in trajectory]
        y = [point[1] for point in trajectory]
        z = [point[2] for point in trajectory]
        
        # Plot the trajectory
        ax.plot(x, y, z, 'r-', label='Ball Trajectory')
        
        # Add a marker for the landing point if it exists
        if landing_point:
            ax.scatter(landing_point[0], landing_point[1], landing_point[2], 
                      color='blue', s=100, label='Landing Point')
        
        # Plot the opponent position
        lateral_reach = self.lateral_reach
        ax.scatter(0, self.court_length, 0, 
                  color='green', s=100, label=f'{opponent} Position')
        
        # Show reach range as a line
        ax.plot([-lateral_reach, lateral_reach], 
                [self.court_length, self.court_length], 
                [0, 0], 'g-', linewidth=2, label=f'{opponent} Reach')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        result_text = "Can Return" if opponent_return_result else "Cannot Return"
        ax.set_title(f'Tennis Serve Trajectory - {opponent} {result_text}')
        
        # Set aspect ratio and limits
        ax.set_box_aspect([1, 2.5, 0.8])
        ax.set_xlim([-self.court_width/2 - 1, self.court_width/2 + 1])
        ax.set_ylim([0, self.court_length + 1])
        ax.set_zlim([0, max(max(z) + 0.5, 3)])
        
        # Add legend
        ax.legend()
        
        return ax

    def _plot_court(self, ax):
        """Plot the tennis court"""
        # Court outline
        ax.plot([-self.court_width/2, -self.court_width/2, self.court_width/2, self.court_width/2, -self.court_width/2],
                [0, self.court_length, self.court_length, 0, 0],
                [0, 0, 0, 0, 0], 'b-')
        
        # Net
        for x in np.linspace(-self.court_width/2, self.court_width/2, 20):
            ax.plot([x, x], [self.court_length/2, self.court_length/2], [0, self.net_height], 'k-', alpha=0.3)
        
        # Center line
        ax.plot([0, 0], 
                [self.court_length/2, self.court_length], 
                [0, 0], 'b-')
        
        # Service line
        ax.plot([-self.court_width/2, self.court_width/2], 
                [self.court_length/2 + self.service_line, self.court_length/2 + self.service_line], 
                [0, 0], 'b-')
        
        # Baseline
        ax.plot([-self.court_width/2, self.court_width/2], 
                [self.court_length, self.court_length], 
                [0, 0], 'b-', linewidth=2)
        
        # Service boxes
        ax.plot([-self.court_width/2, 0], 
                [self.court_length/2 + self.service_line, self.court_length/2 + self.service_line], 
                [0, 0], 'b-')
        ax.plot([0, self.court_width/2], 
                [self.court_length/2 + self.service_line, self.court_length/2 + self.service_line], 
                [0, 0], 'b-')

    def run_simulation(self, height, speed, direction, opponent, serving_side):
        """Run the full serve simulation with given parameters"""
        # Calculate release point and initial velocity with serving side
        release_point = self.calculate_release_point(height, serving_side)
        initial_velocity = self.calculate_initial_velocity(speed, direction, serving_side)
        
        # Simulate trajectory
        trajectory, velocities, times = self.simulate_trajectory(release_point, initial_velocity)
        
        # Check if serve is in
        is_in, landing_point = self.is_serve_in(trajectory)
        
        if not is_in:
            return trajectory, landing_point, False, None, "The serve is OUT!"
        
        # Check if opponent can return
        can_return, explanation = self.can_opponent_return(trajectory, times, opponent, landing_point)
        
        return trajectory, landing_point, is_in, can_return, explanation


class TennisServeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Serve Simulator")
        self.root.geometry("1800x1000")  # Increased window size
        
        # Configure styles
        style = ttk.Style()
        style.configure('Large.TButton', padding=(30, 15), font=('Arial', 14, 'bold'))  # Larger buttons
        style.configure('Large.TLabel', font=('Arial', 11))
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Large.TLabelframe', padding=15)
        style.configure('Large.TLabelframe.Label', font=('Arial', 12, 'bold'))
        
        # Add hover effect for buttons
        style.map('Large.TButton',
                 background=[('active', '#2980b9')],
                 foreground=[('active', 'white')])
        
        self.simulator = TennisServeSimulator()
        self.simulation_running = False
        
        # Create main frame with increased padding
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel with increased width
        input_frame = ttk.LabelFrame(main_frame, text="Serve Settings", style='Large.TLabelframe')
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20), ipadx=30)
        
        # Height input with increased size
        ttk.Label(input_frame, text="Player Height (m):", style='Large.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        self.height_var = tk.StringVar(value="1.85")
        height_scale = ttk.Scale(input_frame, from_=1.50, to=2.20, orient=tk.HORIZONTAL, 
                                length=300, command=lambda x: self.height_var.set(f"{float(x):.2f}"))
        height_scale.set(1.85)
        height_scale.grid(row=0, column=1, sticky=tk.W, pady=(0, 10), padx=(10, 0))
        ttk.Label(input_frame, textvariable=self.height_var, style='Large.TLabel').grid(row=0, column=2, padx=(10, 0))
        
        # Speed input with increased size
        ttk.Label(input_frame, text="Serve Speed:", style='Large.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        self.speed_var = tk.StringVar(value="160.00")
        self.speed_mph = tk.StringVar(value="99.42")  # Move this up before creating the scale
        
        speed_scale = ttk.Scale(input_frame, from_=100.00, to=250.00, orient=tk.HORIZONTAL, 
                              length=300, command=self.update_speed)
        # Don't set scale value until after defining update_speed method
        speed_scale.grid(row=1, column=1, sticky=tk.W, pady=(0, 10), padx=(10, 0))
        
        # Create speed display frame
        speed_frame = ttk.Frame(input_frame)
        speed_frame.grid(row=1, column=2, sticky=tk.W, padx=(10, 0))
        ttk.Label(speed_frame, textvariable=self.speed_var, style='Large.TLabel').pack(side=tk.LEFT)
        ttk.Label(speed_frame, text=" km/h (", style='Large.TLabel').pack(side=tk.LEFT)
        ttk.Label(speed_frame, textvariable=self.speed_mph, style='Large.TLabel').pack(side=tk.LEFT)
        ttk.Label(speed_frame, text=" mph)", style='Large.TLabel').pack(side=tk.LEFT)
        
        # Now set the scale value after everything is initialized
        speed_scale.set(160.00)
        
        # Serving side input with larger combobox
        ttk.Label(input_frame, text="Serving Side:", style='Large.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        self.serving_side_var = tk.StringVar(value="Deuce")
        serving_side_combo = ttk.Combobox(input_frame, textvariable=self.serving_side_var,
                                        values=["Deuce", "Ad"], state="readonly", width=15, font=('Arial', 11))
        serving_side_combo.grid(row=2, column=1, sticky=tk.W, pady=(0, 10), padx=(10, 0))
        
        # Direction input with larger combobox
        ttk.Label(input_frame, text="Serve Direction:", style='Large.TLabel').grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        self.direction_var = tk.StringVar(value="T")
        direction_combo = ttk.Combobox(input_frame, textvariable=self.direction_var, 
                                     values=list(self.simulator.directions.keys()), 
                                     state="readonly", width=15, font=('Arial', 11))
        direction_combo.grid(row=3, column=1, sticky=tk.W, pady=(0, 10), padx=(10, 0))
        
        # Receiver input with larger combobox
        ttk.Label(input_frame, text="Receiver:", style='Large.TLabel').grid(row=4, column=0, sticky=tk.W, pady=(0, 10))
        self.opponent_var = tk.StringVar(value="Receiver")
        opponent_combo = ttk.Combobox(input_frame, textvariable=self.opponent_var, 
                                    values=["Receiver"], state="readonly", width=15, font=('Arial', 11))
        opponent_combo.grid(row=4, column=1, sticky=tk.W, pady=(0, 10), padx=(10, 0))
        
        # Create button frame with increased spacing
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Larger buttons with enhanced styling and more padding
        self.simulate_button = ttk.Button(button_frame, text="▶ Simulate Serve", 
                                        command=self.start_simulation, style='Large.TButton',
                                        width=20)  # Fixed width for consistency
        self.simulate_button.pack(side=tk.LEFT, padx=20)  # More spacing between buttons
        
        self.end_task_button = ttk.Button(button_frame, text="↺ Reset Simulation", 
                                         command=self.end_task, style='Large.TButton',
                                         width=20)  # Fixed width for consistency
        self.end_task_button.pack(side=tk.LEFT, padx=20)
        
        # Stats display with increased size
        stats_frame = ttk.LabelFrame(input_frame, text="Serve Statistics", style='Large.TLabelframe')
        stats_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W+tk.E, pady=(20, 0))
        
        self.stats_text = tk.Text(stats_frame, height=25, width=40, wrap=tk.WORD, 
                                 font=('Arial', 11), padx=10, pady=10)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.insert(tk.END, "Run simulation to see statistics...")
        self.stats_text.config(state=tk.DISABLED)

        # Create right panel for visualization with increased size
        viz_frame = ttk.LabelFrame(main_frame, text="Serve Visualization", style='Large.TLabelframe')
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create the court visualization
        self.fig, self.ax = self.simulator.create_court_visualization()
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create navigation frame for view controls
        navigation_frame = ttk.Frame(viz_frame)
        navigation_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Add camera controls
        self.view_options = ["Front View", "Side View", "Top View", "3D View"]
        self.view_var = tk.StringVar(value="3D View")
        
        view_label = ttk.Label(navigation_frame, text="Camera View:", style='Large.TLabel')
        view_label.pack(side=tk.LEFT, padx=(0, 5))
        
        for view in self.view_options:
            rb = ttk.Radiobutton(navigation_frame, text=view, variable=self.view_var, 
                             value=view, command=self.change_view)
            rb.pack(side=tk.LEFT, padx=5)
        
        # Add home button with larger size
        self.home_button = ttk.Button(navigation_frame, text="⌂ Home View", 
                                    command=self.reset_view, style='Large.TButton',
                                    width=15)  # Slightly smaller than main buttons
        self.home_button.pack(side=tk.RIGHT, padx=20)
        
        # Enhanced status bar
        self.status_var = tk.StringVar(value="Ready to simulate.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, 
                              anchor=tk.W, style='Large.TLabel', padding=(10, 5))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind the window close button
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        """Handle window closing properly"""
        plt.close('all')  # Close all matplotlib figures
        self.root.quit()  # Stop the mainloop
        self.root.destroy()  # Destroy the window
    
    def update_stats(self, text):
        """Update the statistics text widget"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.config(state=tk.DISABLED)
    
    def start_simulation(self):
        """Start the simulation in a separate thread"""
        self.simulation_running = True
        self.simulate_button.config(state=tk.DISABLED)
        self.end_task_button.config(state=tk.NORMAL)
        self.status_var.set("Simulating...")
        
        # Get values from GUI
        height = float(self.height_var.get())
        speed = float(self.speed_var.get()) / 3.6  # Convert km/h to m/s
        direction = self.direction_var.get()
        opponent = self.opponent_var.get()
        
        # Create and start simulation thread
        simulation_thread = threading.Thread(target=self.run_simulation_thread, 
                                          args=(height, speed, direction, opponent))
        simulation_thread.daemon = True
        simulation_thread.start()
    
    def end_task(self):
        """Reset the entire simulation"""
        self.simulation_running = False
        self.simulate_button.config(state=tk.NORMAL)
        
        # Reset input values to defaults
        self.height_var.set("1.85")
        self.speed_var.set("160.00")
        self.direction_var.set("T")
        self.opponent_var.set("Receiver")
        self.serving_side_var.set("Deuce")
        
        # Clear and reset visualization
        self.ax.clear()
        self._plot_empty_court()
        self.canvas.draw()
        
        # Reset stats text
        self.update_stats("Simulation reset. Ready for new simulation...")
        
        # Reset status
        self.status_var.set("Ready to simulate.")

    def _plot_empty_court(self):
        """Plot empty court after termination"""
        self.simulator._plot_court(self.ax)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Tennis Serve Trajectory')
        self.ax.set_box_aspect([1, 2.5, 0.8])
        self.ax.set_xlim([-self.simulator.court_width/2 - 1, self.simulator.court_width/2 + 1])
        self.ax.set_ylim([0, self.simulator.court_length + 1])
        self.ax.set_zlim([0, 3])
    
    def run_simulation_thread(self, height, speed, direction, opponent):
        """Run the simulation in a separate thread"""
        # Run simulation with serving side
        trajectory, landing_point, is_in, can_return, explanation = self.simulator.run_simulation(
            height, speed, direction, opponent, self.serving_side_var.get())
        
        # Update stats text
        stats_text = f"Height: {height:.2f} m\n"
        stats_text += f"Speed: {speed:.2f} m/s ({speed*3.6:.2f} km/h, {speed*self.simulator.MS_TO_MPH:.2f} mph)\n"
        stats_text += f"Direction: {direction}\n"
        stats_text += f"Opponent: {opponent}\n\n"
        
        if not is_in:
            stats_text += "Result: OUT!\n"
            stats_text += "The serve didn't land in the service box."
        else:
            stats_text += "Result: IN!\n"
            stats_text += explanation
            
            # Calculate some additional stats
            distance = np.sqrt(landing_point[0]**2 + landing_point[1]**2)
            stats_text += f"\n\nLanding point: ({landing_point[0]:.2f}, {landing_point[1]:.2f})"
            stats_text += f"\nDistance traveled: {distance:.2f} m"
            
            # Serve effectiveness score (1-10)
            if can_return:
                effectiveness = 5  # Base score for returnable serve
            else:
                effectiveness = 9  # Base score for unreturnable serve
                
            # Adjust based on landing point position
            if abs(landing_point[0]) > 3:  # Wide serve
                effectiveness += 0.5
            if landing_point[1] < self.simulator.court_length/2 + 1:  # Short serve
                effectiveness -= 1
                
            stats_text += f"\nServe effectiveness: {min(10, effectiveness):.1f}/10"
        
        # Update GUI in main thread
        self.root.after(0, self.update_gui, trajectory, landing_point, opponent, 
                        can_return if is_in else False, explanation, stats_text)
    
    def update_gui(self, trajectory, landing_point, opponent, can_return, explanation, stats_text):
        """Update the GUI with simulation results"""
        if not self.simulation_running:
            return
            
        # Update visualization
        self.ax = self.simulator.update_visualization(self.ax, trajectory, landing_point, 
                                                    opponent, can_return, explanation)
        self.canvas.draw()
        
        # Update stats
        self.update_stats(stats_text)
        
        # Update status
        self.status_var.set("Simulation complete.")
        self.simulate_button.config(state=tk.NORMAL)
        self.end_task_button.config(state=tk.DISABLED)

    def reset_view(self):
        """Reset camera to default isometric view"""
        self.ax.view_init(elev=self.simulator.default_elev, 
                         azim=self.simulator.default_azim)
        self.canvas.draw()
        self.view_var.set("3D View")
        
    def change_view(self):
        """Change the camera view of the 3D visualization"""
        view = self.view_var.get()
        
        if view == "Front View":
            self.ax.view_init(elev=0, azim=-90)
        elif view == "Side View":
            self.ax.view_init(elev=0, azim=0)
        elif view == "Top View":
            self.ax.view_init(elev=90, azim=-90)
        elif view == "3D View":
            self.ax.view_init(elev=self.simulator.default_elev, 
                            azim=self.simulator.default_azim)
            
        self.canvas.draw()

    def update_speed(self, value):
        """Update both km/h and mph speed displays"""
        kmh = float(value)
        mph = kmh * 0.621371  # Convert km/h to mph
        self.speed_var.set(f"{kmh:.2f}")
        self.speed_mph.set(f"{mph:.2f}")


def main():
    root = tk.Tk()
    app = TennisServeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()