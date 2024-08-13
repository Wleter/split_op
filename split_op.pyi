from typing import Iterable, Tuple

class Propagation:
    """
    A class representing split opeartor propagation scheme.
    Use :propagate: after setting wave function, time grid and operation stack
    to propagate the initial :WaveFunction: in time specified by :TimeGrid: using operations in :OperationStack:.
    """
    def __init__(self) -> None: ...

    def set_wave_function(self, wave_function: WaveFunction) -> None:
        """
        Set the initial wave function for the propagation
        """

    def set_time_grid(self, time_grid: TimeGrid) -> None:
        """
        Set the time grid for the propagation
        """
    
    def set_operation_stack(self, operation_stack: OperationStack) -> None:
        """
        Set the initial wave function for the propagation
        """
    
    def propagate(self) -> None:
        """
        Propagates the initial wave function in time specified with :TimeGrid: using operations in :OperationStack: 
        """    

    def get_losses(self) -> list[float]:
        """
        Get losses in the appending chronological order and saves losses with LossSaver
        """

    def save_savers(self) -> None:
        """
        Save data accquired by savers during propagation
        """

def gaussian_distribution(x: float, x0: float, sigma: float, momentum: float) -> complex:
    """
    Return gaussian distribution of the form exp(-(x-x0)^2/(2sigma)^2 - i momentum (x - x0)) 
    """


class WaveFunction:
    """
    Creates the wave function from the :array: that was created on the :grids:. 
    It is automaticly normalized to unity.
    """
    def __init__(self, array: Iterable[complex], grids: Iterable[Grid]) -> None: ...

class TimeGrid:
    """
    Creates :TimeGrid: with specified 
    
    :step: for propagation step
    :step_no: number of steps in the propagation
    :im_time: whether time is imaginary, default is False 
    """
    def __init__(self, step: float, step_no: float, im_time: bool = False) -> None: ...

class Grid:
    @staticmethod
    def linear_continous(name: str, start: float, end: float, nodes_no: int, dim_nr: int) -> Grid:
        """
        Creates linear continuos grid with

        :name: specified name
        :start: starting point
        :end: end point
        :nodes_no: number of points
        :dim_nr: number of the dimension, which specifies order in all grids
        """

    @staticmethod
    def linear_countable(name: str, start: float, end: float, nodes_no: int, dim_nr: int) -> Grid:
        """
        Creates linear countable grid with

        :name: specified name
        :start: starting point
        :end: end point
        :nodes_no: number of points
        :dim_nr: number of the dimension, which specifies order in all grids
        """

    @staticmethod
    def custom(name: str, nodes: Iterable[float], weights: Iterable[float], dim_nr: int) -> Grid:
        """
        Creates custom grid with

        :name: specified name
        :nodes: points of the grid
        :weights: weight of the grid for integration
        :dim_nr: number of the dimension, which specifies order in all grids
        """

    def points(self) -> list[float]:
        """
        Returns points of the grid.
        """

    def weights(self) -> list[float]:
        """
        Returns weights of the grid.
        """

    def nodes_no(self) -> int:
        """
        Returns number of points of the grid.
        """

class OperationStack:
    """
    Creates new operation stack. 
    Adding operations to the stack is done for each operations separately.
    """
    def __init__(self) -> None: ...

class FFTTransformation:
    """
    Creates FFT transformation for given :grid:, transformed grid is named :transformed_grid_name:
    """
    def __init__(self, grid: Grid, transformed_grid_name: str) -> None: ...

    def add_operation(self, operation_stack: OperationStack, inverse_second: bool) -> None:
        """
        Add operation to the :operation-stack:, :inverse_second: specifies the order of transformations.
        """

class MatrixTransformation:
    """
    Creates Matrix transformation for given :grid:, which transform it to :transformed_grid:.
    Transformation matrices are added with :set_matrix:
    """
    def __init__(self, grid: Grid, transformed_grid: Grid) -> None: ...

    def set_matrix(self, transformation: Iterable[complex], inverse: Iterable[complex]) -> None:
        """
        Set the transformation matrix and the inverse transformation.
        """

    def add_operation(self, operation_stack: OperationStack, inverse_second: bool) -> None:
        """
        Add operation to the :operation-stack:, :inverse_second: specifies the order of transformations.
        """

    def transformed_grid(self) -> Grid:
        """
        Returns the resulting grid after the transformation. 
        """

class OneDimPropagator:
    """
    Creates one dimensional propagator with given :shape: and the dimension number :dimenion_nr: on which it acts.
    """
    def __init__(self, shape: int, dimension_nr: int) -> None: ...

    def set_operator(self, operator: Iterable[complex]) -> None:
        """
        Set the :operator:.
        """

    def add_operator(self, operator: Iterable[complex]) -> None:
        """
        Add the :operator:, that is combined with other operators.
        """

    def set_loss_checked(self, loss_checked: LossChecker) -> None:
        """
        Set the loss checker :loss_checked: that monitor changes done by this propagator
        """

    def add_operation(self, operation_stack: OperationStack) -> None:
        """
        Add operation to the :operation-stack:.
        """

class NDimPropagator:
    """
    Creates n dimensional propagator.
    """
    def __init__(self) -> None: ...

    def set_operator(self, shape: Iterable[int], operator: Iterable[complex]) -> None:
        """
        Set the :operator: with specified :shape:.
        """

    def add_operator(self, operator: Iterable[complex]) -> None:
        """
        Add the :operator: with specified :shape:, that is combined with other operators.
        """

    def set_loss_checked(self, loss_checked: LossChecker) -> None:
        """
        Set the loss checker that monitor changes done by this propagator
        """

    def add_operation(self, operation_stack: OperationStack) -> None:
        """
        Add operation to the :operation-stack:.
        """

def one_dim_into_propagator(hamiltonian: Iterable[float], grid: Grid, time: TimeGrid, step: str = "half") -> OneDimPropagator:
    """
    Creates 1 dimensional propagator from the :hamiltonian: acting on the :grid: given :TimeGrid: :time: and the :step:
    Resulting operator is exp(-i H dt), where dt depends on :time: and :step: 

    :step: can be either "full" or "half"
    """

def n_dim_into_propagator(shape: Iterable[int], hamiltonian: Iterable[float], time: TimeGrid, step: str = "half") -> NDimPropagator:
    """
    Creates N dimensional propagator from the :hamiltonian: with :shape:, given :TimeGrid: :time: and the :step:
    Resulting operator is exp(-i H dt), where dt depends on :time: and :step: 

    :step: can be either "full" or "half"
    """

def kinetic_hamiltonian(grid: Grid, mass: float, energy: float) -> list[float]:
    """
    Creates kinetic hamiltonian on the :grid:, with :mass: in u units, with :energy: in Kelvin units.
    This form is created on transformed grid using :FFTTranformation:, that have to be applied before this operation.
    The form of the operator is $k^2/2m$
    """

def legendre_transformation(grid: Grid) -> MatrixTransformation:
    """
    Creates legendre transformation given the grid.
    """

def rotational_hamiltonian(radial_grid: Grid, polar_grid: Grid, mass: float, rot_const: float) -> Tuple[list[int], list[float]]:
    """
    Creates rotational Hamiltonian given radial_grid, polar_grid, mass and rotational constant.
    Returns the shape and the data of the Hamiltonian matrix.
    """

class LossChecker:
    """
    Creates new loss checker that checks the loss done by the operator that it is supplied to using :set_loss_checked:.
    Use :new_with_saver: if you want to save loss throughout the propagation.
    """ 
    def __init__(self, name: str) -> None: ...

    """
    Creates new loss checker that has saver that saves the loss throughout the propagation.
    It takes :frames_no: snapshots and saves it to the :filename: and requires time_grid.
    """
    @staticmethod
    def new_with_saver(name: str, frames_no: int, filename: str, time_grid: TimeGrid) -> 'LossChecker': ...

class LeakControl:
    """
    Creates Leak control with given :loss_checker: that monitors and corrects numerical losses.
    """
    def __init__(self, grid: Grid, transformed_grid_name: str) -> None: ...

    def add_operation(self, operation_stack: OperationStack) -> None:
        """
        Add operation to the :operation-stack:.
        """

class WaveFunctionSaver:
    """
    Creates wave function saver for 2d problems, that snapshot wave function probability density during propagation.

    :path: path to save the wave function
    :name: name of the saved file
    :time_grid: used :TimeGrid:
    :x_grid: x grid to save
    :y_grid: y grid to save
    :frames_no: number of frames to take during propagation
    """
    def __init__(self, path: str, name: str, time_grid: TimeGrid, x_grid: Grid, y_grid: Grid, frames_no: int) -> None: ...

    def add_operation(self, operation_stack: OperationStack) -> None:
        """
        Add operation to the :operation-stack:.
        """

class StateSaver:
    """
    Creates state saver, that snapshot the wave function probability density projected on specified :Grid: during propagation.

    :path: path to save the wave function
    :name: name of the saved file
    :time_grid: used :TimeGrid:
    :x_grid: grid on which wave function is projected.
    :frames_no: number of frames to take during propagation
    """
    def __init__(self, path: str, name: str, time_grid: TimeGrid, grid: Grid, frames_no: int) -> None: ...

    def add_operation(self, operation_stack: OperationStack) -> None:
        """
        Add operation to the :operation-stack:.
        """

class BorderDumping:
    """
    Creates border dumping that erase wave function that is on the border.
    Erasing start linearly from mask_width + mask_end from the border and from mask_end it fully vanishes wave function that is there.

    """
    def __init__(self, mask_width: float, mask_end: float, grid: Grid) -> None: ...

    def set_loss_checked(self, loss_checked: LossChecker) -> None:
        """
        Set the loss checker that monitor changes done by this border dumping.
        """

    def add_operation(self, operation_stack: OperationStack) -> None:
        """
        Add operation to the :operation-stack:.
        """