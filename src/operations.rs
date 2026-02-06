use faer::{prelude::c64, Mat};
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{Array1, Array2, ArrayD};
use num::complex::Complex64;
use pyo3::prelude::*;
use cc_constants::units::*;
use split_operator::{border_dumping::{dumping_end, BorderDumping}, control::Apply, hamiltonian_factory::{kinetic_operator, legendre_diagonalization::{associated_legendre_diagonalization_operator, associated_legendre_operator, legendre_diagonalization_operator}, rotational_operator}, leak_control::LeakControl, loss_checker::LossChecker, propagation::OperationStack, propagator::{fft_transformation::FFTTransformation, matrix_transformation::MatrixTransformation, n_dim_propagator::NDimPropagator, non_diagonal_propagator::NonDiagPropagator, one_dim_propagator::OneDimPropagator, propagator_factory, state_matrix_transformation::StateMatrixTransformation, transformation::Order}, time_grid::{select_step, TimeStep}, wave_function_saver::{StateSaver, WaveFunctionSaver}};

use rayon::prelude::*;

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

    pub(crate) fn transformed_grid(&self) -> GridPy {
        GridPy(self.0.grid_transformation.clone())
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

    fn transformed_grid(&self) -> GridPy {
        GridPy(self.0.grid_transformation.clone())
    }
}

#[pyclass(name = "StateMatrixTransformation")]
pub struct StateMatrixTransformationPy(pub StateMatrixTransformation, usize);

#[pymethods]
impl StateMatrixTransformationPy {
    #[new]
    pub(crate) fn new(dimension_nr_dependent: usize, grid: PyRef<GridPy>, transformation_grid: PyRef<GridPy>) -> Self {
        Self(StateMatrixTransformation::new(dimension_nr_dependent, &grid.0, transformation_grid.0.clone()), grid.0.nodes_no)
    }

    pub(crate) fn set_matrices(&mut self, transformations: Vec<Vec<Complex64>>, inverses: Vec<Vec<Complex64>>) {

        let transformations = transformations.iter()
            .map(|x| Array2::from_shape_vec([self.1, self.1], x.clone()).unwrap())
            .collect();
        let inverses = inverses.iter()
            .map(|x| Array2::from_shape_vec([self.1, self.1], x.clone()).unwrap())
            .collect();

        self.0.set_diagonalization_matrices(transformations, inverses)
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>, inverse_second: bool) {
        let order = match inverse_second {
            true => Order::Normal,
            false => Order::InverseFirst,
        };

        operation_stack.0.add_transformation(Box::new(self.0.clone()), order);
    }

    fn transformed_grid(&self) -> GridPy {
        GridPy(self.0.grid_transformation.clone())
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

#[pyclass(name = "NonDiagPropagator")]
pub struct NonDiagPropagatorPy(pub NonDiagPropagator);

#[pymethods]
impl NonDiagPropagatorPy {
    #[new]
    pub(crate) fn new(dimension_nr: usize) -> Self {
        Self(NonDiagPropagator::new(dimension_nr))
    }

    pub(crate) fn set_operators(&mut self, dim_size: usize, operators: Vec<Vec<Complex64>>) {
        let shape = [dim_size, dim_size];

        let operators = operators.into_iter()
            .map(|x| Array2::from_shape_vec(shape, x).unwrap())
            .collect();

        self.0.set_operators(operators)
    }

    pub(crate) fn set_loss_checked(&mut self, loss_checked: PyRef<LossCheckerPy>) {
        self.0.set_loss_checked(loss_checked.0.clone());
    }

    pub(crate) fn add_operation(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        operation_stack.0.add_propagator(Box::new(self.0.clone()));
    }

    #[staticmethod]
    pub(crate) fn get_coriolis(
        r_grid: PyRef<GridPy>, 
        j_grid: PyRef<GridPy>, 
        omega_grid: PyRef<GridPy>,
        mass_u: f64,
        j_tot: usize,
        time_grid: PyRef<TimeGridPy>,
        step: &str,
    ) -> Self {
        // use only those grids don't use with any other grid
        assert!(r_grid.0.dimension_no == 0);
        assert!(j_grid.0.dimension_no == 1);
        assert!(omega_grid.0.dimension_no == 2);

        let step = match step {
            "full" => TimeStep::Full,
            "half" => TimeStep::Half,
            _ => panic!("wrong time step {step}")
        };
        let time_step = select_step(step, &time_grid.0);

        let omega_points = &omega_grid.0.nodes;
        let omega_dim = omega_grid.0.nodes_no;
        let j_points = &j_grid.0.nodes;
        let r_points = &r_grid.0.nodes;

        let coriolis_spatial: Vec<f64> = r_points.iter()
            .map(|&r| - 1. / (2. * mass_u * Dalton::TO_BASE * r * r))
            .collect();

        let c_matrices: Vec<Array2<Complex64>> = j_points.par_iter()
            .map(|&j_value| Array2::from_shape_fn((omega_dim, omega_dim), |(i, j)| {
                    unsafe {
                        let k_left = *omega_points.get_unchecked(i);
                        let k_right = *omega_points.get_unchecked(j);

                        if k_left == k_right + 1. || k_right == k_left + 1. {
                            if j_value * (j_value + 1.) < k_left * k_right 
                                || j_tot as f64 * (j_tot as f64 + 1.) < k_left * k_right
                            {
                                Complex64::ZERO
                            } else {
                                Complex64::from(f64::sqrt((j_value * (j_value + 1.0f64) - k_left * k_right) 
                                    * ((j_tot * (j_tot + 1)) as f64 - k_left * k_right)))
                            }
                        } else {
                            Complex64::ZERO
                        }
                    }
                })
            )
            .collect();

        let mut exponents = Vec::with_capacity(coriolis_spatial.len() * c_matrices.len());
        for spatial in coriolis_spatial {
            let exp: Vec<Array2<Complex64>> = c_matrices.par_iter()
                .map(|c| {
                    let factor: c64 = -c64::i() * spatial * c64::from(time_step);

                    let faer_c = c.view().into_faer_complex();
                    let eigen = faer_c.selfadjoint_eigendecomposition(faer::Side::Upper);

                    let exp_diag = eigen.s()
                        .column_vector()
                        .iter()
                        .map(|&x| (x * factor).exp())
                        .collect::<Vec<c64>>();

                    let exp_mat = Mat::from_fn(exp_diag.len(), exp_diag.len(), |i, j| {
                        if i == j {
                            unsafe {
                                *exp_diag.get_unchecked(i)
                            }
                        } else {
                            c64::from(0.)
                        }
                    });

                    let values = eigen.u() * exp_mat * eigen.u().adjoint();

                    values.as_ref().into_ndarray_complex().to_owned()
                })
                .collect();
            exponents.extend(exp);
        }

        let mut non_diag = NonDiagPropagator::new(omega_grid.0.dimension_no);
        non_diag.set_operators(exponents);

        NonDiagPropagatorPy(non_diag)
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

#[pyfunction(signature = (shape, hamiltonian, time, step="half"))]
pub fn complex_n_dim_into_propagator(shape: Vec<usize>, hamiltonian: Vec<Complex64>, time: PyRef<TimeGridPy>, step: &str) -> NDimPropagatorPy {
    let hamiltonian = ArrayD::from_shape_vec(shape, hamiltonian).unwrap();

    let step = match step {
        "full" => TimeStep::Full,
        "half" => TimeStep::Half,
        _ => panic!("wrong time step {step}")
    };

    NDimPropagatorPy(propagator_factory::complex_n_dim_into_propagator(hamiltonian, &time.0, step))
}

#[pyfunction]
pub fn kinetic_hamiltonian(grid: PyRef<GridPy>, mass_u: f64) -> Vec<f64> {
    kinetic_operator::kinetic_hamiltonian(&grid.0, mass_u * Dalton::TO_BASE).to_vec()
}

#[pyfunction]
pub fn legendre_transformation(grid: PyRef<GridPy>) -> MatrixTransformationPy {
    MatrixTransformationPy(legendre_diagonalization_operator(&grid.0), grid.0.nodes_no)
}

#[pyfunction]
pub fn associated_legendre_transformation(grid: PyRef<GridPy>, omega: isize) -> MatrixTransformationPy {
    MatrixTransformationPy(associated_legendre_diagonalization_operator(&grid.0, omega), grid.0.nodes_no)
}

#[pyfunction]
pub fn associated_legendre_transformations(grid: PyRef<GridPy>, omega_grid: PyRef<GridPy>) -> StateMatrixTransformationPy {
    StateMatrixTransformationPy(associated_legendre_operator(&grid.0, &omega_grid.0), grid.0.nodes_no)
}

#[pyfunction]
pub fn rotational_hamiltonian(radial_grid: PyRef<GridPy>, polar_grid: PyRef<GridPy>, mass_u: f64, rot_const: f64) -> (Vec<usize>, Vec<f64>) {
    let array = rotational_operator::rotational_hamiltonian(&radial_grid.0, &polar_grid.0, mass_u * Dalton::TO_BASE, rot_const);

    let shape = array.shape().to_vec();
    let (v, _) = array.into_raw_vec_and_offset();

    (shape, v)
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
    pub(crate) fn new(name: &str, time_grid: PyRef<TimeGridPy>, x_grid: PyRef<GridPy>, y_grid: PyRef<GridPy>, frames_no: usize) -> Self {
        WaveFunctionSaverPy(WaveFunctionSaver::new(name.to_string(), &time_grid.0, &x_grid.0, &y_grid.0, frames_no))
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
    pub(crate) fn new(name: &str, time_grid: PyRef<TimeGridPy>, state_grid: PyRef<GridPy>, frames_no: usize) -> Self {
        StateSaverPy(StateSaver::new(name.to_string(), &time_grid.0, &state_grid.0, frames_no))
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