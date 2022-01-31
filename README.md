# FL_Converge

We provide the code implementation for paper [On the Convergence of Heterogeneous Federated Learning with Arbitrary Adaptive Online Model Pruning](https://arxiv.org/abs/2201.11803) 

The implementation is based opensource implementation of [FedAvg](https://github.com/shaoxiongji/federated-learning)

## REQUIREMENTS

python>=3.6
pytorch>=0.4

# Train on your own

## Example to run weights Pruning on mnist iid
> python main_fed_Pweight_AvgAll.py --dataset mnist  --num_users 100 --frac 0.1 --num_channels 1 --model mlp --iid

For training with all settings including.
```
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 4, 4, 4, 4],
        [1, 1, 1, 1, 1, 4, 4, 4, 4, 7],
        [1, 1, 1, 1, 4, 4, 4, 4, 4, 4],
        [1, 1, 1, 1, 1, 1, 3, 4, 7, 7],
        [1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 1, 1, 1, 2, 3, 4, 4, 4, 4],
        [1, 1, 1, 1, 4, 4, 4, 4, 7, 7],
        [1, 1, 1, 1, 2, 3, 4, 5, 6, 7],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        [1, 1, 1, 1, 5, 5, 6, 6, 7, 7],
        [1, 1, 1, 4, 5, 5, 6, 6, 7, 7],
        [1, 2, 3, 4, 5, 5, 6, 6, 7, 7],
        [1, 4, 5, 5, 6, 6, 6, 7, 7, 7],
        [1, 4, 4, 4, 7, 7, 7, 7, 7, 7],
        [2, 2, 3, 3, 4, 4, 5, 6, 7, 7]
````

You may add, remove, or select partial of settings by modifying the setting_array.

# Pretrained Results
Goto `DemoData` folder and run any notebook file. 

You may need to change some imput names to generate other figures. Or read the csv file on your own.



Note: Under development. It can reproduce the results presented in the paper, however it is quite messy now and may not be suitable to be used as foundation for other works.

Note: The scripts will be slow without the implementation of parallel computing. 

# FLOPs Calculation

Use the notebook file in `FLOPsCount` file to calculate a model's FLOPs and memory usage.

## Ackonwledgements
Acknowledgements give to [shaoxiongji](https://github.com/shaoxiongji) for the original FedAvg implementation.
