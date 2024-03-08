# sml-practical
Group project for Oxford University SB 2.2 Statistical Machine Learning.
Assignment: [SMLAssessedPractical.pdf](https://github.com/tobias-schnabel/sml-practical/files/14511487/SMLAssessedPractical.pdf)

## Setup
1. To activate local environment for the first time, in terminal use
```bash
conda env create -f environment.yml
```
2. After, upon launching IDE run
```bash
conda activate sml-practical-env
```
3. And set IDE interpreter to sml-practical-env

4. To update the environment, deactivate and use 
```bash
 conda env update --name sml-practical-env --file environment.yml --prune
```