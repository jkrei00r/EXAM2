# region imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
# endregion

# region function definitions
def S(x):
    """
    Computes the Fresnel integral S(x) = ∫_0^x sin(t^2) dt using quad.

    Parameters:
    x (float): Upper limit of the integral.

    Returns:
    float: The value of the Fresnel integral S(x).
    """
    s, _ = quad(lambda t: np.sin(t**2), 0, x)
    return s

def Exact(x):
    """
    Computes the exact solution of the initial value problem at x.

    Parameters:
    x (float): The value at which to compute the exact solution.

    Returns:
    float: The exact solution y(x) = 1 / (2.5 - S(x)) + 0.01 * x**2.
    """
    return 1 / (2.5 - S(x)) + 0.01 * x**2

def ODE_System(x, y):
    """
    Defines the system of ODEs for the initial value problem.

    Parameters:
    x (float): The independent variable.
    y (list): The state variables (in this case, y[0] is y(x)).

    Returns:
    list: The derivative of the state variable(s).
    """
    Y = y[0]  # Rename the state variable for convenience
    Ydot = (Y - 0.01 * x**2)**2 * np.sin(x**2) + 0.02 * x  # Calculate the derivative
    return [Ydot]

def Plot_Result(*args):
    """
    Plots the numerical and exact solutions according to the specified formatting criteria.

    Parameters:
    *args: Contains the plottable arrays for the numerical and exact solutions.
    """
    xRange_Num, y_Num, xRange_Xct, y_Xct = args  # Unpack the arguments

    plt.figure(figsize=(10, 6))
    plt.plot(xRange_Xct, y_Xct, label='Exact', linestyle='-', color='blue')
    plt.plot(xRange_Num, y_Num, label='Numerical', marker='^', linestyle='', color='red')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("IVP: y'=(y-0.01x^2)^2 sin(x^2)+0.02x, y(0)=0.4")
    plt.legend()

    plt.xlim(0.0, 6.0)
    plt.ylim(0.0, 1.0)

    plt.xticks(np.arange(0.0, 6.1, 0.2))
    plt.yticks(np.arange(0.0, 1.1, 0.1))

    plt.gca().xaxis.set_tick_params(direction='in', top=True)
    plt.gca().yaxis.set_tick_params(direction='in', right=True)

    plt.grid(True)
    plt.show()

def main():
    """
    This function solves the initial value problem of problem 1 of exam 2, Spring 2023.
    y'=(y-0.01x**2)**2*sin(x**2)+0.02x
    y(0)=0.4
    It then plots the numerical solution and the exact solution according to the formatting criteria.
    """
    xRange = np.arange(0, 5.2, 0.2)  # Create a numpy array for the x range to evaluate numerical solution (h=0.2)
    xRange_xct = np.linspace(0, 5, 500)  # Create a numpy array for the x range for the exact solution
    Y0 = [0.4]  # Initial condition y(0) = 0.4

    sln = solve_ivp(ODE_System, [0, 5], Y0, t_eval=xRange)  # Numerically solve i.v.p. with default RK45 method
    xctSln = np.array([Exact(x) for x in xRange_xct])  # Produce array of y values for exact solution

    Plot_Result(xRange, sln.y[0], xRange_xct, xctSln)  # Call the plotting function to produce the required plot
# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion