import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

class RealisticHeartSimulation:
    """
    1NAT - Realistic 3D Heartbeat Simulation
    Physiologically accurate cardiac cycle with 3D visualization
    """
    
    def __init__(self, heart_rate=72, dt=0.02):
        self.heart_rate = heart_rate  # Normal resting HR
        self.dt = dt
        self.t = 0
        
        # Cardiac cycle phases (physiologically accurate)
        self.cycle_duration = 60.0 / heart_rate  # seconds per beat
        self.systole_duration = 0.3  # seconds (ventricular contraction)
        self.diastole_duration = self.cycle_duration - self.systole_duration
        
        # Hemodynamic parameters (realistic values)
        self.stroke_volume = 70  # mL per beat
        self.cardiac_output = (self.stroke_volume * heart_rate) / 1000  # L/min
        self.ejection_fraction = 0.55  # 55% (normal)
        
        # Blood pressure (mmHg)
        self.systolic_pressure = 120
        self.diastolic_pressure = 80
        self.current_pressure = self.diastolic_pressure
        
        # Heart volumes (mL)
        self.end_diastolic_volume = 120  # EDV
        self.end_systolic_volume = 50    # ESV
        self.current_volume = self.end_diastolic_volume
        
        # Organs with realistic blood flow distribution
        self.organs = {
            'Brain': {'flow_percent': 14, 'resistance': 0.8, 'perfusion': 100, 'color': (1.0, 0.8, 0.8)},
            'Heart': {'flow_percent': 4, 'resistance': 1.0, 'perfusion': 100, 'color': (1.0, 0.5, 0.5)},
            'Kidneys': {'flow_percent': 22, 'resistance': 0.7, 'perfusion': 100, 'color': (0.8, 0.5, 0.3)},
            'Lungs': {'flow_percent': 100, 'resistance': 0.2, 'perfusion': 100, 'color': (1.0, 0.7, 0.8)},
            'Liver': {'flow_percent': 27, 'resistance': 0.85, 'perfusion': 100, 'color': (0.6, 0.3, 0.2)},
            'Muscles': {'flow_percent': 20, 'resistance': 1.2, 'perfusion': 100, 'color': (0.9, 0.6, 0.6)},
            'GI_Tract': {'flow_percent': 24, 'resistance': 0.9, 'perfusion': 100, 'color': (0.8, 0.7, 0.5)}
        }
        
        # ECG simulation parameters
        self.ecg_value = 0
        
        # Data history
        self.time_history = []
        self.pressure_history = []
        self.volume_history = []
        self.ecg_history = []
        self.max_history = 300
        
    def get_cardiac_phase(self):
        """Determine current phase of cardiac cycle"""
        cycle_time = self.t % self.cycle_duration
        
        if cycle_time < self.systole_duration:
            # Systole (ventricular contraction)
            phase_progress = cycle_time / self.systole_duration
            return 'systole', phase_progress
        else:
            # Diastole (ventricular filling)
            phase_progress = (cycle_time - self.systole_duration) / self.diastole_duration
            return 'diastole', phase_progress
    
    def calculate_pressure(self, phase, progress):
        """Calculate realistic blood pressure waveform"""
        if phase == 'systole':
            # Rapid pressure rise during systole
            pressure = self.diastolic_pressure + \
                      (self.systolic_pressure - self.diastolic_pressure) * \
                      np.sin(progress * np.pi / 2) ** 2
        else:
            # Exponential decay during diastole
            pressure = self.systolic_pressure - \
                      (self.systolic_pressure - self.diastolic_pressure) * \
                      (1 - np.exp(-3 * progress))
        return pressure
    
    def calculate_volume(self, phase, progress):
        """Calculate ventricular volume"""
        if phase == 'systole':
            # Volume decreases during ejection
            volume = self.end_diastolic_volume - \
                    (self.end_diastolic_volume - self.end_systolic_volume) * progress
        else:
            # Volume increases during filling
            volume = self.end_systolic_volume + \
                    (self.end_diastolic_volume - self.end_systolic_volume) * progress
        return volume
    
    def simulate_ecg(self, phase, progress):
        """Simulate ECG waveform (P-QRS-T complex)"""
        cycle_time = self.t % self.cycle_duration
        normalized_time = cycle_time / self.cycle_duration
        
        # P wave (atrial depolarization)
        if 0 <= normalized_time < 0.1:
            return 0.3 * np.sin(normalized_time * 10 * np.pi)
        # QRS complex (ventricular depolarization)
        elif 0.15 <= normalized_time < 0.25:
            t_qrs = (normalized_time - 0.15) / 0.1
            if t_qrs < 0.3:
                return -0.5 * np.sin(t_qrs * 10 * np.pi)
            else:
                return 2.0 * np.sin((t_qrs - 0.3) * 10 * np.pi)
        # T wave (ventricular repolarization)
        elif 0.4 <= normalized_time < 0.6:
            return 0.5 * np.sin((normalized_time - 0.4) * 5 * np.pi)
        else:
            return 0
    
    def update_organ_perfusion(self, pressure):
        """Update organ perfusion based on cardiac output"""
        mean_arterial_pressure = self.diastolic_pressure + \
                                (self.systolic_pressure - self.diastolic_pressure) / 3
        
        for organ_name, organ_data in self.organs.items():
            # Calculate blood flow based on pressure and resistance
            flow_rate = (pressure / mean_arterial_pressure) * \
                       organ_data['flow_percent'] * self.cardiac_output
            
            # Update perfusion (smooth transition)
            target_perfusion = (flow_rate / organ_data['flow_percent']) * 100
            organ_data['perfusion'] += 0.1 * (target_perfusion - organ_data['perfusion'])
    
    def update_simulation(self):
        """Update simulation by one time step"""
        phase, progress = self.get_cardiac_phase()
        
        # Update hemodynamic parameters
        self.current_pressure = self.calculate_pressure(phase, progress)
        self.current_volume = self.calculate_volume(phase, progress)
        self.ecg_value = self.simulate_ecg(phase, progress)
        
        # Update organ perfusion
        self.update_organ_perfusion(self.current_pressure)
        
        # Store history
        self.time_history.append(self.t)
        self.pressure_history.append(self.current_pressure)
        self.volume_history.append(self.current_volume)
        self.ecg_history.append(self.ecg_value)
        
        if len(self.time_history) > self.max_history:
            self.time_history.pop(0)
            self.pressure_history.pop(0)
            self.volume_history.pop(0)
            self.ecg_history.pop(0)
        
        self.t += self.dt
        return phase, progress
    
    def create_3d_heart(self, ax, scale=1.0):
        """Create realistic 3D heart model"""
        ax.clear()
        
        # Create heart shape using spherical coordinates
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        # Parametric heart surface
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Modified heart equations for 3D
        x = scale * (16 * np.sin(u_grid)**3)
        y = scale * (13 * np.cos(u_grid) - 5 * np.cos(2*u_grid) - 
                     2 * np.cos(3*u_grid) - np.cos(4*u_grid))
        z = scale * 10 * np.sin(v_grid) * np.sin(u_grid)
        
        # Normalize
        x = x / 20
        y = y / 20
        z = z / 20
        
        # Get current phase for coloring
        phase, progress = self.get_cardiac_phase()
        
        # Color based on oxygenation and contraction
        if phase == 'systole':
            # Bright red during contraction
            color_intensity = 0.7 + 0.3 * progress
            color = (color_intensity, 0.1, 0.1)
        else:
            # Darker red during relaxation
            color_intensity = 0.5 + 0.2 * (1 - progress)
            color = (color_intensity, 0.2, 0.2)
        
        # Plot heart surface
        surf = ax.plot_surface(x, y, z, color=color, alpha=0.9,
                              edgecolor='darkred', linewidth=0.3,
                              shade=True, antialiased=True)
        
        # Add chambers (ventricles) as spheres
        chamber_scale = 0.3 * scale
        
        # Left ventricle
        u_c = np.linspace(0, 2 * np.pi, 20)
        v_c = np.linspace(0, np.pi, 20)
        u_c, v_c = np.meshgrid(u_c, v_c)
        x_lv = chamber_scale * np.sin(v_c) * np.cos(u_c) - 0.2
        y_lv = chamber_scale * np.sin(v_c) * np.sin(u_c) - 0.1
        z_lv = chamber_scale * np.cos(v_c)
        
        ax.plot_surface(x_lv, y_lv, z_lv, color=(0.4, 0.0, 0.0), alpha=0.6)
        
        # Right ventricle
        x_rv = chamber_scale * 0.8 * np.sin(v_c) * np.cos(u_c) + 0.2
        y_rv = chamber_scale * 0.8 * np.sin(v_c) * np.sin(u_c) - 0.1
        z_rv = chamber_scale * 0.8 * np.cos(v_c)
        
        ax.plot_surface(x_rv, y_rv, z_rv, color=(0.3, 0.0, 0.1), alpha=0.6)
        
        # Set labels and limits
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        # Add title with phase
        phase_text = "SYSTOLE (Contraction)" if phase == 'systole' else "DIASTOLE (Filling)"
        ax.set_title(f'3D Heart Model - {phase_text}', 
                    fontsize=11, fontweight='bold', pad=10)
        
        # Rotate for better view
        ax.view_init(elev=20, azim=self.t * 20)  # Slow rotation
        
    def draw_hemodynamics(self, ax):
        """Draw pressure-volume loop and waveforms"""
        ax.clear()
        
        if len(self.time_history) < 2:
            return
        
        # Plot pressure waveform
        ax.plot(self.time_history, self.pressure_history, 
               'r-', linewidth=2.5, label='Arterial Pressure')
        
        # Mark current point
        ax.plot(self.time_history[-1], self.pressure_history[-1], 
               'ro', markersize=10, zorder=5)
        
        # Reference lines
        ax.axhline(y=120, color='orange', linestyle='--', 
                  alpha=0.5, linewidth=1.5, label='Systolic (120)')
        ax.axhline(y=80, color='blue', linestyle='--', 
                  alpha=0.5, linewidth=1.5, label='Diastolic (80)')
        
        ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Pressure (mmHg)', fontsize=10, fontweight='bold')
        ax.set_title('Arterial Blood Pressure', fontsize=12, fontweight='bold')
        ax.set_xlim(max(0, self.t - 4), self.t + 0.5)
        ax.set_ylim(60, 140)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='upper left', fontsize=8)
        
        # Add current value box
        current_p = self.pressure_history[-1]
        ax.text(0.98, 0.95, f'{current_p:.0f} mmHg', 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', 
                        alpha=0.8, edgecolor='black', linewidth=2))
    
    def draw_ecg(self, ax):
        """Draw ECG waveform"""
        ax.clear()
        
        if len(self.time_history) < 2:
            return
        
        ax.plot(self.time_history, self.ecg_history, 
               'g-', linewidth=2, label='ECG')
        
        ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Voltage (mV)', fontsize=10, fontweight='bold')
        ax.set_title('Electrocardiogram (ECG)', fontsize=12, fontweight='bold')
        ax.set_xlim(max(0, self.t - 4), self.t + 0.5)
        ax.set_ylim(-1, 2.5)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.axhline(y=0, color='black', linewidth=1)
        
        # Mark waves
        ax.text(0.05, 0.95, 'P-QRS-T Complex', 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    def draw_organs_3d(self, ax):
        """Draw 3D organ perfusion"""
        ax.clear()
        
        # Arrange organs in 3D space around heart
        positions = [
            (0, 0, 1.5),      # Brain (top)
            (0, 0, 0),        # Heart (center)
            (-1.2, 0, -0.3),  # Kidneys (back left)
            (1.2, 0, 0.5),    # Lungs (front right)
            (-1.2, 0, 0.5),   # Liver (front left)
            (0, 1.2, -0.3),   # Muscles (right)
            (0, -1.2, -0.3),  # GI Tract (left)
        ]
        
        organ_names = list(self.organs.keys())
        
        for i, (organ_name, pos) in enumerate(zip(organ_names, positions)):
            organ_data = self.organs[organ_name]
            perfusion = organ_data['perfusion']
            
            # Color based on perfusion
            base_color = np.array(organ_data['color'])
            intensity = perfusion / 100
            color = base_color * intensity
            color = np.clip(color, 0, 1)
            
            # Draw organ as sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            u, v = np.meshgrid(u, v)
            
            size = 0.35
            x = size * np.sin(v) * np.cos(u) + pos[0]
            y = size * np.sin(v) * np.sin(u) + pos[1]
            z = size * np.cos(v) + pos[2]
            
            ax.plot_surface(x, y, z, color=color, alpha=0.8, 
                           edgecolor='black', linewidth=0.5)
            
            # Label
            ax.text(pos[0], pos[1], pos[2], organ_name.replace('_', ' '),
                   fontsize=8, fontweight='bold', ha='center')
            
            # Blood vessel to heart (pulsing)
            phase, progress = self.get_cardiac_phase()
            alpha = 0.3 + 0.5 * progress if phase == 'systole' else 0.3
            ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 
                   'r-', linewidth=2, alpha=alpha)
        
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-1, 2])
        ax.set_title('3D Organ System', fontsize=11, fontweight='bold')
        ax.view_init(elev=20, azim=self.t * 15)
    
    def draw_cardiac_parameters(self, ax):
        """Display cardiac parameters"""
        ax.clear()
        ax.axis('off')
        
        phase, progress = self.get_cardiac_phase()
        
        params_text = f"""
╔══════════════════════════════════════╗
║     CARDIAC PARAMETERS               ║
╚══════════════════════════════════════╝

⚡ VITAL SIGNS:
  • Heart Rate: {self.heart_rate} BPM
  • Blood Pressure: {self.current_pressure:.0f} mmHg
  • Stroke Volume: {self.stroke_volume} mL
  • Cardiac Output: {self.cardiac_output:.1f} L/min
  • Ejection Fraction: {self.ejection_fraction*100:.0f}%

💓 CURRENT STATE:
  • Phase: {phase.upper()}
  • Progress: {progress*100:.0f}%
  • Volume: {self.current_volume:.0f} mL
  • Time: {self.t:.2f} s

🏥 ORGAN PERFUSION:
"""
        
        for organ_name, organ_data in self.organs.items():
            perfusion = organ_data['perfusion']
            status = "●" if perfusion > 80 else "◐" if perfusion > 60 else "○"
            params_text += f"  {status} {organ_name:12s}: {perfusion:5.1f}%\n"
        
        ax.text(0.05, 0.95, params_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', 
                        alpha=0.8, edgecolor='black', linewidth=2))


def run_realistic_simulation():
    """Run the realistic 3D heart simulation"""
    print("="*60)
    print("1NAT - REALISTIC 3D HEARTBEAT SIMULATION")
    print("Physiologically Accurate Cardiac Model")
    print("="*60)
    print("\nInitializing realistic heart model...")
    print("Close window to stop simulation.\n")
    
    sim = RealisticHeartSimulation(heart_rate=72, dt=0.02)
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Realistic 3D Heartbeat-Driven Human Body Simulation', 
                 fontsize=16, fontweight='bold')
    
    # Create subplots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')  # 3D Heart
    ax2 = fig.add_subplot(2, 3, 2)                    # Blood Pressure
    ax3 = fig.add_subplot(2, 3, 3)                    # ECG
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')  # 3D Organs
    ax5 = fig.add_subplot(2, 3, 5)                    # Parameters
    ax6 = fig.add_subplot(2, 3, 6)                    # Reserved
    ax6.axis('off')
    
    def animate(frame):
        """Animation update"""
        phase, progress = sim.update_simulation()
        
        # Heart scale based on volume
        scale = 0.8 + 0.4 * (sim.current_volume / sim.end_diastolic_volume)
        
        sim.create_3d_heart(ax1, scale=scale)
        sim.draw_hemodynamics(ax2)
        sim.draw_ecg(ax3)
        sim.draw_organs_3d(ax4)
        sim.draw_cardiac_parameters(ax5)
        
        if frame % 50 == 0:
            print(f"[{sim.t:6.2f}s] Phase: {phase:8s} | "
                  f"BP: {sim.current_pressure:5.1f} mmHg | "
                  f"Vol: {sim.current_volume:5.1f} mL")
        
        return ax1, ax2, ax3, ax4, ax5
    
    anim = FuncAnimation(fig, animate, frames=None, 
                        interval=20, blit=False, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)


if __name__ == "__main__":
    run_realistic_simulation()
