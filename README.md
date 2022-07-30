# LCLS Cu Injector Model for ML Testing
Working repo for the LCLS Cu line injector NN surrogate model for ML testing

## Environment
```bash
$ conda env create -f injmodel.yml
```
or
```bash
$ conda conda create --name injmodel --file injmodel.txt
```

## Basic Usage
A [3 variable BO example](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/injector_emit_prediction_BO_3var_pytorch_example.ipynb) shows how to optimize emittance values from the pytorch version of the injector surrogate. 

A [9 variable BO example](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/injector_emit_prediction_BO_9var_pytorch_example.ipynb) shows how to optimize emittance*bmag values calculated from the pytorch version of the injector surrogate. 

[9 variable BO comparisons](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/injector_emit_prediction_BO_comparisons_9var_pytorch.ipynb) runs trials for BOs with various prior means and collects data for [performance comparison](https://github.com/sylvia5monthes/lcls_cu_injector_ml_model/blob/47486ecf454a0bb2088b07fc1163f94eba20fbc1/injector_surrogate/BO_comparison_visualization.ipynb) 

For more details on the injector tuning and the capabilties of the model, see the [lcls_cu_injector_tuning repo](https://github.com/slaclab/lcls_cu_injector_tuning/).
