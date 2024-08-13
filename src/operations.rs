use ndarray::{Array1, Array2, ArrayD};
use num::complex::Complex64;
use pyo3::prelude::*;
use quantum::{particle::Particle, particles::Particles, units::{energy_units::{Energy, Kelvin}, mass_units::{Dalton, Mass}}};
use split_operator::{border_dumping::{dumping_end, BorderDumping}, control::Apply, hamiltonian_factory::kinetic_operator, leak_control::LeakControl, loss_checker::LossChecker, propagation::OperationStack, propagator::{fft_transformation::FFTTransformation, matrix_transformation::MatrixTransformation, n_dim_propagator::NDimPropagator, one_dim_propagator::OneDimPropagator, propagator_factory, transformation::Order}, time_grid::TimeStep, wave_function_saver::{StateSaver, WaveFunctionSaver}};

use crate::{GridPy, TimeGridPy};

#[pyclass(name = "OperationStack")]
pub struct OperationStackPy(pub OperationStack);

#[pymethods]
impl OperationStackPy {
    #[new]
    pub(crate) fn new() -> Self {
        Self(OperationStack::new())
    }
}

#[pyclass(name = "FFTTransformation")]
pub struct FFTTransformationPy(pub FFTTransformation);

#[pymethods]
impl FFTTransformationPy {
    #[new]
    pub(crate) fn new(grid: PyRef<GridPy>, transformed_grid_name: &str) -> Self {
        Self(FFTTransformation::new(&grid.0, transformed_grid_name))
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>, inverse_second: bool) {
        let order = match inverse_second {
            true => Order::Normal,
            false => Order::InverseFirst,
        };

        operation_stack.0.add_transformation(Box::new(self.0.clone()), order);
    }
}

#[pyclass(name = "MatrixTransformation")]
pub struct MatrixTransformationPy(pub MatrixTransformation, usize);

#[pymethods]
impl MatrixTransformationPy {
    #[new]
    pub(crate) fn new(grid: PyRef<GridPy>, transformation_grid: PyRef<GridPy>) -> Self {
        Self(MatrixTransformation::new(&grid.0, transformation_grid.0.clone()), grid.0.nodes_no)
    }

    pub(crate) fn set_matrix(&mut self, transformation: Vec<Complex64>, inverse: Vec<Complex64>) {
        let transformation = Array2::from_shape_vec([self.1, self.1], transformation).unwrap();
        let inverse = Array2::from_shape_vec([self.1, self.1], inverse).unwrap();

        self.0.set_diagonalization_matrix(transformation, inverse)
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>, inverse_second: bool) {
        let order = match inverse_second {
            true => Order::Normal,
            false => Order::InverseFirst,
        };

        operation_stack.0.add_transformation(Box::new(self.0.clone()), order);
    }
}


#[pyclass(name = "OneDimPropagator")]
pub struct OneDimPropagatorPy(pub OneDimPropagator);

#[pymethods]
impl OneDimPropagatorPy {
    #[new]
    pub(crate) fn new(shape: usize, dimension_nr: usize) -> Self {
        Self(OneDimPropagator::new(shape, dimension_nr))
    }

    pub(crate) fn set_operator(&mut self, operator: Vec<Complex64>) {
        let operator = Array1::from_vec(operator);
        self.0.set_operator(operator)
    }

    pub(crate) fn add_operator(&mut self, operator: Vec<Complex64>) {
        let operator = Array1::from_vec(operator);
        self.0.add_operator(operator)
    }

    pub(crate) fn set_loss_checked(&mut self, loss_checked: PyRef<LossCheckerPy>) {
        self.0.set_loss_checked(loss_checked.0.clone());
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        operation_stack.0.add_propagator(Box::new(self.0.clone()));
    }
}


#[pyclass(name = "NDimPropagator")]
pub struct NDimPropagatorPy(pub NDimPropagator);

#[pymethods]
impl NDimPropagatorPy {
    #[new]
    pub(crate) fn new() -> Self {
        Self(NDimPropagator::new())
    }

    pub(crate) fn set_operator(&mut self, shape: Vec<usize>, operator: Vec<Complex64>) {
        let operator = ArrayD::from_shape_vec(shape, operator).unwrap();

        self.0.set_operator(operator)
    }

    pub(crate) fn add_operator(&mut self, shape: Vec<usize>, operator: Vec<Complex64>) {
        let operator = ArrayD::from_shape_vec(shape, operator).unwrap();
        self.0.add_operator(operator)
    }

    pub(crate) fn set_loss_checked(&mut self, loss_checked: PyRef<LossCheckerPy>) {
        self.0.set_loss_checked(loss_checked.0.clone());
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        operation_stack.0.add_propagator(Box::new(self.0.clone()));
    }
}

#[pyfunction(signature = (hamiltonian, grid, time, step="half"))]
pub fn one_dim_into_propagator(hamiltonian: Vec<f64>, grid: PyRef<GridPy>, time: PyRef<TimeGridPy>, step: &str) -> OneDimPropagatorPy {
    let hamiltonian = Array1::from_vec(hamiltonian);

    let step = match step {
        "full" => TimeStep::Full,
        "half" => TimeStep::Half,
        _ => panic!("wrong time step {step}")
    };

    OneDimPropagatorPy(propagator_factory::one_dim_into_propagator(hamiltonian, &grid.0, &time.0, step))
}

#[pyfunction(signature = (shape, hamiltonian, time, step="half"))]
pub fn n_dim_into_propagator(shape: Vec<usize>, hamiltonian: Vec<f64>, time: PyRef<TimeGridPy>, step: &str) -> NDimPropagatorPy {
    let hamiltonian = ArrayD::from_shape_vec(shape, hamiltonian).unwrap();

    let step = match step {
        "full" => TimeStep::Full,
        "half" => TimeStep::Half,
        _ => panic!("wrong time step {step}")
    };

    NDimPropagatorPy(propagator_factory::n_dim_into_propagator(hamiltonian, &time.0, step))
}

#[pyfunction]
pub fn kinetic_hamiltonian(grid: PyRef<GridPy>, mass: f64, energy: f64) -> Vec<f64> {
    let particle = Particle::new("emulate", Mass(2.0 * mass, Dalton));

    let particles = Particles::new_pair(particle.clone(), particle, Energy(energy, Kelvin));

    kinetic_operator::kinetic_hamiltonian(&grid.0, &particles).to_vec()
}

#[pyclass(name = "LossChecker")]
pub struct LossCheckerPy(pub LossChecker);

#[pymethods]
impl LossCheckerPy {
    #[new]
    pub(crate) fn new(name: &str) -> Self {
        LossCheckerPy(LossChecker::new(name))
    }

    #[staticmethod]
    pub(crate) fn new_with_saver(name: &str, frames_no: usize, filename: String, time_grid: PyRef<TimeGridPy>) -> Self {
        LossCheckerPy(LossChecker::new_with_saver(name, frames_no, filename, &time_grid.0))
    }
}

#[pyclass(name = "LeakControl")]
pub struct LeakControlPy(pub LeakControl);

#[pymethods]
impl LeakControlPy {
    #[new]
    pub(crate) fn new(loss_checker: PyRefMut<LossCheckerPy>) -> Self {
        let mut leak_control = LeakControl::new();
        leak_control.add_loss_checker(loss_checker.0.clone());

        LeakControlPy(leak_control)
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        operation_stack.0.add_control(Box::new(self.0.clone()), Apply::FirstHalf | Apply::SecondHalf);
    }
}

#[pyclass(name = "WaveFunctionSaver")]
pub struct WaveFunctionSaverPy(pub WaveFunctionSaver);

#[pymethods]
impl WaveFunctionSaverPy {
    #[new]
    pub(crate) fn new(path: &str, name: &str, time_grid: PyRef<TimeGridPy>, x_grid: PyRef<GridPy>, y_grid: PyRef<GridPy>, frames_no: usize) -> Self {
        WaveFunctionSaverPy(WaveFunctionSaver::new(path.to_string(), name.to_string(), &time_grid.0, &x_grid.0, &y_grid.0, frames_no))
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        operation_stack.0.add_saver(Box::new(self.0.clone()), Apply::FirstHalf);
    }
}

#[pyclass(name = "StateSaver")]
pub struct StateSaverPy(pub StateSaver);

#[pymethods]
impl StateSaverPy {
    #[new]
    pub(crate) fn new(path: &str, name: &str, time_grid: PyRef<TimeGridPy>, state_grid: PyRef<GridPy>, frames_no: usize) -> Self {
        StateSaverPy(StateSaver::new(path.to_string(), name.to_string(), &time_grid.0, &state_grid.0, frames_no))
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        operation_stack.0.add_saver(Box::new(self.0.clone()), Apply::FirstHalf);
    }
}

#[pyclass(name = "BorderDumping")]
pub struct BorderDumpingPy(pub BorderDumping);

#[pymethods]
impl BorderDumpingPy {
    #[new]
    pub(crate) fn new(mask_width: f64, mask_end: f64, grid: PyRef<GridPy>) -> Self {
        BorderDumpingPy(BorderDumping::new(dumping_end(mask_width, mask_end, &grid.0), &grid.0))
    }

    pub(crate) fn set_loss_checked(&mut self, loss_checked: PyRef<LossCheckerPy>) {
        self.0.add_loss_checker(loss_checked.0.clone());
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        operation_stack.0.add_control(Box::new(self.0.clone()), Apply::FirstHalf);
    }
}