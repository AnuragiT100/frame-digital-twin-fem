import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----- FEM Classes -----

class Node:
    def __init__(self, x, y, fixed=[False, False, False]):
        self.x = x
        self.y = y
        self.fixed = fixed

class Element:
    def __init__(self, node_i, node_j, E, A, I):
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.A = A
        self.I = I
        self.L = self.compute_length()
        self.k_global = self.compute_stiffness()

    def compute_length(self):
        return np.sqrt((self.node_j.x - self.node_i.x)**2 + (self.node_j.y - self.node_i.y)**2)

    def compute_stiffness(self):
        L = self.L
        E, A, I = self.E, self.A, self.I
        c = (self.node_j.x - self.node_i.x) / L
        s = (self.node_j.y - self.node_i.y) / L

        k_local = E / L * np.array([
            [ A/L,      0,          0,      -A/L,      0,          0      ],
            [ 0,   12*I/L**2,   6*I/L,       0,   -12*I/L**2,   6*I/L     ],
            [ 0,    6*I/L,     4*I,          0,    -6*I/L,      2*I      ],
            [-A/L,      0,          0,       A/L,      0,          0      ],
            [ 0,  -12*I/L**2,  -6*I/L,       0,    12*I/L**2,  -6*I/L     ],
            [ 0,    6*I/L,     2*I,          0,    -6*I/L,      4*I      ],
        ])

        T = np.zeros((6,6))
        T[0,0] = c;  T[0,1] = s
        T[1,0] = -s; T[1,1] = c
        T[2,2] = 1
        T[3,3] = c;  T[3,4] = s
        T[4,3] = -s; T[4,4] = c
        T[5,5] = 1

        return T.T @ k_local @ T

class FrameModel:
    def __init__(self, E, A, I):
        self.nodes = []
        self.elements = []
        self.E = E
        self.A = A
        self.I = I
        self.dof_per_node = 3

    def add_node(self, x, y, fixed=[False, False, False]):
        self.nodes.append(Node(x, y, fixed))

    def add_element(self, n1_idx, n2_idx):
        node_i = self.nodes[n1_idx]
        node_j = self.nodes[n2_idx]
        element = Element(node_i, node_j, self.E, self.A, self.I)
        self.elements.append(element)

    def assemble_global_stiffness(self):
        n_dof = len(self.nodes) * self.dof_per_node
        K = np.zeros((n_dof, n_dof))

        for elem in self.elements:
            dof_map = []
            n1 = self.nodes.index(elem.node_i)
            n2 = self.nodes.index(elem.node_j)
            dof_map.extend([n1*self.dof_per_node, n1*self.dof_per_node+1, n1*self.dof_per_node+2,
                            n2*self.dof_per_node, n2*self.dof_per_node+1, n2*self.dof_per_node+2])

            for i in range(6):
                for j in range(6):
                    K[dof_map[i], dof_map[j]] += elem.k_global[i,j]
        return K

    def apply_boundary_conditions(self, K, F):
        fixed_dofs = []
        for i, node in enumerate(self.nodes):
            for dof_i, fixed in enumerate(node.fixed):
                if fixed:
                    fixed_dofs.append(i*self.dof_per_node + dof_i)

        free_dofs = np.setdiff1d(np.arange(len(F)), fixed_dofs)

        K_reduced = K[np.ix_(free_dofs, free_dofs)]
        F_reduced = F[free_dofs]

        return K_reduced, F_reduced, free_dofs

    def solve(self, loads):
        n_dof = len(self.nodes) * self.dof_per_node
        F = np.zeros(n_dof)
        for dof_idx, val in loads.items():
            F[dof_idx] = val

        K = self.assemble_global_stiffness()
        K_reduced, F_reduced, free_dofs = self.apply_boundary_conditions(K, F)
        d_reduced = np.linalg.solve(K_reduced, F_reduced)

        d_full = np.zeros(n_dof)
        d_full[free_dofs] = d_reduced
        return d_full

# ----- Sensor Simulation & Anomaly Detection -----

def simulate_sensors(frame_model, displacements, noise_level=1e-5):
    dof_per_node = frame_model.dof_per_node
    disp_sensors = []
    for i, node in enumerate(frame_model.nodes):
        ux = displacements[i*dof_per_node]
        uy = displacements[i*dof_per_node + 1]
        ux += np.random.normal(0, noise_level)
        uy += np.random.normal(0, noise_level)
        disp_sensors.append((ux, uy))

    strain_sensors = []
    for elem in frame_model.elements:
        n1 = frame_model.nodes.index(elem.node_i)
        n2 = frame_model.nodes.index(elem.node_j)
        L = elem.L
        ux1 = displacements[n1*dof_per_node]
        uy1 = displacements[n1*dof_per_node +1]
        ux2 = displacements[n2*dof_per_node]
        uy2 = displacements[n2*dof_per_node +1]

        x1 = elem.node_i.x + ux1
        y1 = elem.node_i.y + uy1
        x2 = elem.node_j.x + ux2
        y2 = elem.node_j.y + uy2
        length_deformed = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        strain = (length_deformed - L)/L + np.random.normal(0, noise_level*10)
        strain_sensors.append(strain)

    return disp_sensors, strain_sensors

def detect_anomalies(disp_sensors, strain_sensors, disp_thresh=0.02, strain_thresh=0.001):
    for ux, uy in disp_sensors:
        if abs(ux) > disp_thresh or abs(uy) > disp_thresh:
            return True
    for strain in strain_sensors:
        if abs(strain) > strain_thresh:
            return True
    return False

# ----- Animation -----

def animate_digital_twin(frame_model, total_frames=300):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,6))

    # Frame deformation plot
    ax1.set_title("Frame Deformation (exaggerated)")
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 5)
    ax1.set_aspect('equal')

    # Mid-node vertical displacement over time
    ax2.set_title("Mid-Node Vertical Displacement Over Time")
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-0.002, 0.002)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vertical Displacement (m)")

    # Anomaly detection status
    ax3.set_title("Anomaly Detection Status")
    ax3.axis('off')
    status_text = ax3.text(0.5, 0.5, 'Normal', ha='center', va='center', fontsize=20, color='green')

    time_data = []
    disp_data = []

    # Plot undeformed frame in dashed black lines
    for elem in frame_model.elements:
        x = [elem.node_i.x, elem.node_j.x]
        y = [elem.node_i.y, elem.node_j.y]
        ax1.plot(x, y, 'k--', linewidth=1)

    # Lines to update
    line_def, = ax1.plot([], [], 'r-', linewidth=2)
    line_disp, = ax2.plot([], [], 'b-')

    dof_per_node = frame_model.dof_per_node
    total_dof = len(frame_model.nodes) * dof_per_node
    scale = 1000  # exaggeration factor for deformation visualization

    def update(frame):
        time = frame * 0.1
        load_magnitude = 20000 * (1 + 0.5 * np.sin(0.5 * time))  # oscillating load

        # Solve displacements
        displacements = frame_model.solve({2*dof_per_node + 1: -load_magnitude})

        # Deformed coordinates (scaled)
        deformed_x = [node.x + scale * displacements[i*dof_per_node] for i, node in enumerate(frame_model.nodes)]
        deformed_y = [node.y + scale * displacements[i*dof_per_node + 1] for i, node in enumerate(frame_model.nodes)]

        xs = []
        ys = []
        for elem in frame_model.elements:
            idx1 = frame_model.nodes.index(elem.node_i)
            idx2 = frame_model.nodes.index(elem.node_j)
            xs.extend([deformed_x[idx1], deformed_x[idx2], None])
            ys.extend([deformed_y[idx1], deformed_y[idx2], None])
        line_def.set_data(xs, ys)

        # Update mid-node vertical displacement graph
        time_data.append(time)
        disp_mid_node = displacements[2*dof_per_node + 1]  # vertical displacement node 2
        disp_data.append(disp_mid_node)
        if len(time_data) > 200:
            time_data.pop(0)
            disp_data.pop(0)
        line_disp.set_data(time_data, disp_data)
        ax2.set_xlim(max(0, time-20), time+0.1)

        # Sensor simulation & anomaly detection
        disp_sensors, strain_sensors = simulate_sensors(frame_model, displacements)
        anomaly = detect_anomalies(disp_sensors, strain_sensors)

        if anomaly:
            status_text.set_text("ANOMALY DETECTED!")
            status_text.set_color('red')
        else:
            status_text.set_text("Normal")
            status_text.set_color('green')

        return line_def, line_disp, status_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)
    plt.tight_layout()

    ani.save('frame_animation.gif', writer='pillow', fps=10)
    plt.show()

# ===== MAIN =====
if __name__ == "__main__":
    E = 50e9
    A = 0.005
    I = 5e-5

    model = FrameModel(E, A, I)
    model.add_node(0, 0, fixed=[True, True, True])
    model.add_node(5, 0, fixed=[True, True, True])
    model.add_node(5, 4, fixed=[False, False, False])

    model.add_element(0, 2)
    model.add_element(1, 2)

    animate_digital_twin(model)



