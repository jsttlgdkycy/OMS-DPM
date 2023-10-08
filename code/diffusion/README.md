# OMS-DPM: Optimizing the Model Schedule for Diffusion Probabilistic Models

The official code for the paper [OMS-DPM: Optimizing the Model Schedule for Diffusion Probabilistic Models](https://arxiv.org/abs/2306.08860) (**ICML 2023**) by Enshu Liu*, Xuefei Ning*, Zinan Lin*, Huazhong Yang, and Yu Wang. OMS-DPM provides a method of using multi-model sampling in the timestep dimension, as well as a search algorithm for optimizing the model schedule.

--------------------

# Code Examples

We offer some examples of using OMS-DPM to 
- run diffusion sample using searched our model schedules
- use our trained predictor checkpoints to search for model schedules
- train predictors using our datasets of model schedule and performance 

## Sample images using searched model schedules

We support using model schedule in diffusion in './code/diffusion/examples/ddpm_and_guided_diffusion'. Our trained models are available at https://drive.google.com/drive/folders/1GBzFNgpSqrvBx9wbt7XtBHyAH3WQTrFU?usp=drive_link. We recommend to put these pre-trained models for FID calculation in './model_zoo' to avoid extra path modifications. We provide some of our searched model schedules at https://drive.google.com/drive/folders/1aozGx30ncBcVKEiRXh2QLAfs7Kab56BF?usp=drive_link. Edit the 'sampling.model_schedule.load' in configs in './code/diffusion/examples/ddpm_and_guided_diffusion/configs' to the path of the model schedule before running with the following command.


```
python ./code/diffusion/examples/ddpm_and_guided-diffusion/main.py --config ./code/diffusion/examples/ddpm_and_guided-diffusion/configs/celeba.yml --sample_type dpmsolver --sample --fid --use_model_schedule
```

We recommend to use the FID statistics at https://drive.google.com/drive/folders/1GBzFNgpSqrvBx9wbt7XtBHyAH3WQTrFU?usp=drive_link to reproduce the results in our paper.

## Use trained predictor to search for model schedules

We provide some predictor checkpoints at https://drive.google.com/drive/folders/1TVpAyMvBxRHpleVg1WkcBWIWD5HTvT-e?usp=drive_link. We recommend to put these predictor checkpoints in './model_zoo' to avoid extra path modifications. The following command is an example.

```
python ./code/predictor/main_predictor.py --type search --resume ./predictor_checkpoints/predictor_cifar10_dpm_solver.pth --budget 4000 --config predictor_cifar10_dpm_solver.yml
```

The search process will produce 'final_population.pth' file which contains the final population of the evolutionary search algorithm. Use the following command to evaluate the search results.

```
python ./code/diffusion/examples/ddpm_and_guided-diffusion/main.py --config ./code/diffusion/examples/ddpm_and_guided-diffusion/configs/cifar10.yml --sample_type dpmsolver --sample --fid --use_model_schedule --gpu 0 --load_population <PATH TO THE FINAL POPULATION FILE>
```

## Train predictors

We provide datasets of model schedule and performance pair at https://drive.google.com/drive/folders/1fgJn4ZqWxOJ4Hq16F0NXbTFnfDYdaFl3?usp=drive_link. Edit the dataset.path in configs in './code/predictor/configs' and train a predictor using the following command.

'''
python ./code/predictor/main_predictor.py --config <CONFIG NAME>
'''

--------------------

# Acknowledgement

This repository is heavily based on https://github.com/LuChengTHU/dpm-solver/tree/main/examples/ddpm_and_guided-diffusion and https://github.com/ermongroup/ddim. We thank for these valuable works.

--------------------

# References

If you find the code useful for your research, please consider citing
```bib
@article{lu2022dpm,
  title={DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps},
  author={Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2206.00927},
  year={2022}
}
```