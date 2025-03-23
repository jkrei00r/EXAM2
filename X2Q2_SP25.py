# region imports
from scipy.integrate import solve_ivp
from math import sin
import numpy as np
from matplotlib import pyplot as plt
# endregion

# region class definitions
class circuit():
    def __init__(self, R=10, L=20, C=0.05, A=20, w=20, p=0):
        '''
        Initializes the circuit object with given parameters.

        :param R: Resistance in ohms (Ω)
        :param L: Inductance in henries (H)
        :param C: Capacitance in farads (F)
        :param A: Amplitude of the voltage source in volts (V)
        :param w: Angular frequency of the voltage source in radians per second (rad/s)
        :param p: Phase of the voltage source in radians (rad)
        '''
        # region attributes
        self.R = R
        self.L = L
        self.C = C
        self.A = A
        self.w = w
        self.p = p
        self.t = None
        self.X = None
        # endregion

    # region methods
    def ode_system(self, t, X):
        """
        Defines the system of ODEs for the two-loop RLC circuit.

        :param t: Current time in seconds.
        :param X: State variables [i1, i2, vc].
        :return: List of derivatives of state variables [di1/dt, di2/dt, dvc/dt].
        """
        i1, i2, vc = X
        # Voltage source v(t) = A * sin(w*t + p)
        v_t = self.A * sin(self.w * t + self.p)
        # Derivatives
        di1_dt = (v_t - self.R * (i1 - i2)) / self.L  # KVL for left loop
        di2_dt = (v_t - self.R * (i1 - i2)) / self.L - i2 / (self.R * self.C)  # KVL for right loop
        dvc_dt = -i2 / self.C  # Capacitor relationship
        return [di1_dt, di2_dt, dvc_dt]

    def simulate(self, t=10, pts=500):
        """
        Simulates the transient behavior of the circuit.

        :param t: Total simulation time in seconds.
        :param pts: Number of points in the simulation.
        :return: None. Stores time and state variables in attributes.
        """
        self.t = np.linspace(0, t, pts)
        X0 = [0, 0, 0]  # Initial conditions: i1=0, i2=0, vc=0
        sol = solve_ivp(self.ode_system, [0, t], X0, t_eval=self.t)
        self.X = sol.y  # Store the state variables

    def doPlot(self, ax=None):
        """
        Plots the currents i1 and i2 over time, and the voltage vc over time on a secondary y-axis.

        :param ax: Matplotlib axis object (optional). If None, creates a new plot.
        :return: None.
        """
        if ax is None:
            fig, ax = plt.subplots()
            QTPlotting = False  # Using CLI and showing the plot
        else:
            QTPlotting = True

        # Extract state variables
        i1 = self.X[0]
        i2 = self.X[1]
        vc = self.X[2]

        # Plot currents on the left y-axis
        ax.plot(self.t, i1, label='i1(t) (A)', color='blue')
        ax.plot(self.t, i2, label='i2(t) (A)', color='green')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True)

        # Create a secondary y-axis for voltage
        ax2 = ax.twinx()
        ax2.plot(self.t, vc, label='vc(t) (V)', color='red')
        ax2.set_ylabel('Voltage (V)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')  # Moved legend to upper left

        ax.set_title('RLC Circuit Transient Response: Currents and Voltage')

        if not QTPlotting:
            plt.show()
    # endregion
# endregion

# region function definitions
def main():
    """
    Main function to solve problem 2 on the exam.
    :return: None.
    """
    # Create a circuit object with the given default values
    Circuit = circuit(R=10, L=20, C=0.05, A=20, w=20, p=0)

    # Simulate and plot with the default values
    print("Displaying plot with default values:")
    print(f"R = {Circuit.R} Ω, L = {Circuit.L} H, C = {Circuit.C} F, v(t) = {Circuit.A}·sin({Circuit.w}·t + {Circuit.p})")
    Circuit.simulate(t=10, pts=500)
    Circuit.doPlot()

    # Ask if the user wants to input additional values
    goAgain = True
    while goAgain:
        response = input("Do you want to input additional values and simulate again? (yes/no): ").strip().lower()
        if response != 'yes':
            goAgain = False
        else:
            # Solicit user input for circuit parameters
            R = float(input("Enter resistance R (Ω): "))
            L = float(input("Enter inductance L (H): "))
            C = float(input("Enter capacitance C (F): "))
            A = float(input("Enter amplitude of voltage source A (V): "))
            w = float(input("Enter angular frequency w (rad/s): "))
            p = float(input("Enter phase p (rad): "))

            # Update circuit parameters
            Circuit.R = R
            Circuit.L = L
            Circuit.C = C
            Circuit.A = A
            Circuit.w = w
            Circuit.p = p

            # Simulate and plot with the new values
            Circuit.simulate(t=10, pts=500)
            Circuit.doPlot()
# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion