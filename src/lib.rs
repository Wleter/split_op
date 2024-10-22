use std::mem::take;

use ndarray::ArrayD;
use num::complex::Complex64;
use operations::OperationStackPy;
use pyo3::prelude::*;
use split_operator::{grid::Grid, propagation::Propagation, time_grid::TimeGrid, wave_function::WaveFunction};

use operations::*;

mod operations;

#[pyclass(name = "Propagation")]
struct PropagationPy(Propagation);

#[pymethods]
impl PropagationPy {
    #[new]
    pub fn new() -> Self {
        Self(Propagation::default())
    }

    pub fn set_wave_function(&mut self, wave_function: PyRef<WaveFunctionPy>) {
        self.0.set_wave_function(wave_function.0.to_owned())
    }

    pub fn set_time_grid(&mut self, time_grid: PyRef<TimeGridPy>) {
        self.0.set_time_grid(time_grid.0.to_owned())
    }

    pub fn set_operation_stack(&mut self, mut operation_stack: PyRefMut<OperationStackPy>) {
        self.0.set_operation_stack(take(&mut operation_stack.0));
    }

    pub fn propagate(&mut self) {
        self.0.propagate();
    }

    pub fn get_losses(&mut self) -> Vec<f64> {
        self.0.print_losses();
        
        self.0.get_losses()
    }

    pub fn save_savers(&mut self) {
        self.0.savers_save()
    }

    pub fn time_grid(&self) -> TimeGridPy {
        TimeGridPy(self.0.time_grid().clone())
    }

    pub fn wave_function(&self) -> WaveFunctionPy {
        WaveFunctionPy(self.0.wave_function().clone())
    }
}


#[pyclass(name = "WaveFunction")]
struct WaveFunctionPy(WaveFunction);

#[pymethods]
impl WaveFunctionPy {
    /// creates new wave function with given data, it is then normalized
    #[new]
    fn init(array: Vec<Complex64>, grids: Vec<PyRef<GridPy>>) -> Self {
        let grids: Vec<Grid> = grids.into_iter()
            .map(|x| x.0.clone())
            .collect();

        let shape: Vec<usize> = grids.iter()
            .map(|x| x.nodes_no)
            .collect();

        let array = array.into_iter()
            .map(|x| x)
            .collect();
        let array = ArrayD::from_shape_vec(shape, array).unwrap();

        let mut wave = WaveFunctionPy(WaveFunction::new(array, grids));
        wave.0.normalize(1.0);
        wave
    }
}

#[pyfunction]
fn gaussian_distribution(x: f64, x0: f64, sigma: f64, momentum: f64) -> Complex64 {
    split_operator::wave_function::gaussian_distribution(x, x0, sigma, momentum)
}

#[pyclass(name = "TimeGrid")]
struct TimeGridPy(TimeGrid); 

#[pymethods]
impl TimeGridPy {
    #[new]
    #[pyo3(signature = (step, step_no, im_time = false))]
    fn init(step: f64, step_no: usize, im_time: bool) -> Self {
        TimeGridPy(TimeGrid { step, step_no, im_time })
    }

    fn step(&self) -> f64 {
        self.0.step
    }

    fn step_no(&self) -> usize {
        self.0.step_no
    }

    fn im_time(&self) -> bool {
        self.0.im_time
    }
}

#[pyclass(name = "Grid")]
struct GridPy(Grid); 

#[pymethods]
impl GridPy {
    #[staticmethod]
    fn linear_continuos(name: &str, start: f64, end: f64, nodes_no: usize, dim_nr: usize) -> GridPy {
        GridPy(Grid::new_linear_continuos(name, start, end, nodes_no, dim_nr))
    }
    
    #[staticmethod]
    fn linear_countable(name: &str, start: f64, end: f64, nodes_no: usize, dim_nr: usize) -> GridPy {
        GridPy(Grid::new_linear_countable(name, start, end, nodes_no, dim_nr))
    }

    #[staticmethod]
    fn custom(name: &str, nodes: Vec<f64>, weights: Vec<f64>, dim_nr: usize) -> GridPy {
        GridPy(Grid::new_custom(name, nodes, weights, dim_nr))
    }

    fn points(&self) -> Vec<f64> {
        self.0.nodes.clone()
    }

    fn weights(&self) -> Vec<f64> {
        self.0.weights.clone()
    }

    fn nodes_no(&self) -> usize {
        self.0.nodes_no
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn split_op(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(gaussian_distribution, m)?)?;

    m.add_class::<GridPy>()?;
    m.add_class::<TimeGridPy>()?;
    m.add_class::<WaveFunctionPy>()?;
    m.add_class::<PropagationPy>()?;

    m.add_function(wrap_pyfunction!(one_dim_into_propagator, m)?)?;
    m.add_function(wrap_pyfunction!(n_dim_into_propagator, m)?)?;
    m.add_function(wrap_pyfunction!(complex_n_dim_into_propagator, m)?)?;
    m.add_function(wrap_pyfunction!(kinetic_hamiltonian, m)?)?;
    m.add_function(wrap_pyfunction!(rotational_hamiltonian, m)?)?;
    m.add_function(wrap_pyfunction!(legendre_transformation, m)?)?;
    m.add_function(wrap_pyfunction!(associated_legendre_transformation, m)?)?;
    m.add_function(wrap_pyfunction!(associated_legendre_transformations, m)?)?;

    m.add_class::<OperationStackPy>()?;

    m.add_class::<FFTTransformationPy>()?;
    m.add_class::<MatrixTransformationPy>()?;
    m.add_class::<StateMatrixTransformationPy>()?;

    m.add_class::<OneDimPropagatorPy>()?;
    m.add_class::<NDimPropagatorPy>()?;
    m.add_class::<NonDiagPropagatorPy>()?;

    m.add_class::<LossCheckerPy>()?;
    m.add_class::<LeakControlPy>()?;
    m.add_class::<WaveFunctionSaverPy>()?;
    m.add_class::<StateSaverPy>()?;
    m.add_class::<BorderDumpingPy>()?;

    Ok(())
}


#[cfg(test)]
mod tests {
    use faer::{mat, prelude::c64, Mat};
    use faer_ext::*;

    use ndarray::arr2;
    use quantum::units::{energy_units::Kelvin, Unit};
    use split_operator::hamiltonian_factory::analytic_potentials::lennard_jones;

    use crate::*;

    #[test]
    fn ground_state() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let mut propagation = PropagationPy::new();

            let x_start = 8.0;
            let x_end = 20.0;
            let x_no = 512;

            let grid = Bound::new(py, GridPy::linear_continuos("space", x_start, x_end, x_no, 0)).unwrap();

            let energy = 1e-7;
            let mass = 6.0;
            let momentum = (2.0f64 * mass * energy * Kelvin::TO_AU_MUL).sqrt();

            let mut wave_function_array = vec![Complex64::ZERO; x_no];
            for (i, x) in grid.borrow().points().iter().enumerate() {
                wave_function_array[i] = gaussian_distribution(*x, 14.0, 2.0, momentum);
            }
            let wave_function = Bound::new(py, WaveFunctionPy::init(wave_function_array, vec![grid.borrow()])).unwrap();

            let time_grid = Bound::new(py, TimeGridPy::init(50.0, 1000, true)).unwrap();

            let r6 = 9.7;
            let d6 = 0.0003;
            let mut potential_array = vec![0.0; x_no];
            for (i, x) in grid.borrow().points().iter().enumerate() {
                potential_array[i] = lennard_jones(*x, d6, r6);
            }

            let mut potential_propagator = one_dim_into_propagator(
                potential_array,
                grid.borrow(),
                time_grid.borrow(),
                "half"
            );

            let kinetic_array = kinetic_hamiltonian(grid.borrow(), mass, energy);
            let mut kinetic_propagator = one_dim_into_propagator(
                kinetic_array,
                grid.borrow(),
                time_grid.borrow(),
                "full",
            );
            let mut fft_transform = FFTTransformationPy::new(grid.borrow(), "momentum");

            let mut wave_function_saver = StateSaverPy::new(
                "data/lj_ground",
                time_grid.borrow(),
                grid.borrow(),
                120,
            );

            let loss_checker = Bound::new(py, LossCheckerPy::new("leak control")).unwrap();
            let mut leak_control = LeakControlPy::new(loss_checker.borrow_mut());

            let operation_stack = Bound::new(py, OperationStackPy::new()).unwrap();
            leak_control.add_operation(operation_stack.borrow_mut());
            wave_function_saver.add_operation(operation_stack.borrow_mut());
            potential_propagator.add_operation(operation_stack.borrow_mut());
            fft_transform.add_operation(operation_stack.borrow_mut(), true);
            kinetic_propagator.add_operation(operation_stack.borrow_mut());

            propagation.set_wave_function(wave_function.borrow());
            propagation.set_time_grid(time_grid.borrow());
            propagation.set_operation_stack(operation_stack.borrow_mut());

            println!("{:.2e}", propagation.0.wave_function().array);

            propagation.propagate();

            println!("{:.2e}", propagation.0.wave_function().array);
        });
    }

    #[test]
    fn test_exponent() {
        let value = arr2(&[[Complex64::ONE, Complex64::I],
                            [-Complex64::I, -Complex64::ONE]]);

        let faer_view = value.view().into_faer_complex();
        let eigen = faer_view.selfadjoint_eigendecomposition(faer::Side::Upper);

        let exp: Vec<c64> = eigen.s().column_vector().iter()
            .map(|x| x.exp())
            .collect();

        let exp = Mat::from_fn(exp.len(), exp.len(), |i, j| 
            if i == j {
                exp[i]
            } else {
                0.0.into()
            }
        );

        let exp = eigen.u() * exp * eigen.u().adjoint();

        let expected = mat![[c64::from(3.54648), 1.3683 * c64::i()],
                            [-1.3683 * c64::i(), c64::from(0.809885)]];

        assert!((exp - expected).norm_max() < 1e-5);
    }
}
