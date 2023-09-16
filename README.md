# HAM
Code for "Improving adversarial transferability through hybrid augmentation".

## Requirements

- python 3.8
- torch 1.8
- pretrainedmodels 0.7
- numpy 1.19
- pandas 1.2

## Implementation

- **Prepare models**

  Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model), which are converted from widely used Tensorflow models. Then put these models into `./models/`

- **Generate adversarial examples**

 You can run this HAM as following:
  
  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python HAM.py --output_dir outputs
  ```
  where `gpuid` can be set to any free GPU ID in your machine. And adversarial examples will be generated in directory `./outputs`.
  
- **Evaluations**

  Running `verify.py` to evaluate the attack success rate.
  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python verify.py
  ```

    ## Acknowledgments ##

Code refers to [SSA](https://github.com/yuyang-long/SSA)