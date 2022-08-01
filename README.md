# LCLS Cu Injector Model for ML Testing
Working repo for the LCLS Cu line injector NN surrogate model for ML testing

## Environment
```bash
$ conda env create -f baxenv.yml
```
or
```bash
$ conda conda create --name baxenv --file baxenv.txt
```

## Basic Usage
- A [3 variable BO example](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/injector_emit_prediction_BO_3var_pytorch_example.ipynb) shows how to optimize emittance values from the [pytorch version of the injector surrogate](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/a1ff4a70c573692304d8741461408cdbce9eced9/injector_surrogate/pytorch_injector_surrogate_model.py). 

- A [9 variable BO example](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/injector_emit_prediction_BO_9var_pytorch_example.ipynb) shows how to optimize emittance*bmag values calculated from the pytorch version of the injector surrogate. 

- [9 variable BO comparisons](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/injector_emit_prediction_BO_comparisons_9var_pytorch.ipynb) runs trials for BOs with various prior means and collects performance comparison data (saved in /results) that can be visualized in this [comparison visualization example](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/BO_comparison_visualization.ipynb). Plots are saved in /BO-plots. The NN models used as prior means are trained [here](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/main/injector_surrogate/BO_pytorch_9var_NN_priors.ipynb) and are saved alongside their transformers in /results. 

### [Results](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/tree/main/injector_surrogate/results) Terminology 
- **grid<num_samples_per_variable>^9.pt** stores the <num_samples_per_variable>^9 meshgrid of -emittance*bmag outputs used to train NN prior mean models.
- **model<num_samples_per_variable>\_<num_hidden_layers>hidden\_<num_nodes_per_hidden_layer>nodes\_<num_epochs>epoch\_<learning_rate>\_<cutoff_value>** where the model is trained on samples where outputs are in the range [cutoff_value, 0] or [-20, 0] if cutoff_value is not specified. 
- **transformer_y_<num_samples_per_variable>^9_<cutoff_value>** corresponds to models that have the same num_samples_per_variable and cutoff_value. 
- **surr_const_<model_names**>.pt**: BO comparison performance data where the BO runs with the following prior means are being compared

For more details on the injector tuning and the capabilties of the model, see the [lcls_cu_injector_tuning repo](https://github.com/slaclab/lcls_cu_injector_tuning/).
