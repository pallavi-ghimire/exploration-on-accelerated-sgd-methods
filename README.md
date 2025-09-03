# Exploration of Accelerated Stochastic Gradient Methods
This repository contains the implementation and experiments for my MSc Data Science thesis on Exploration of Accelerated Stochastic Gradient Methods for Least Squares Problems. It includes Python scripts for different optimizers (SGD, SVRG, ASG), synthetic dataset generation, real-world data experiments, and utilities for configuration, analysis, and plotting.

## requirements.txt
Python dependencies required for running the experiments.

## Dataset Generation
There already is a sample real-world dataset obtained from Kaggle: https://www.kaggle.com/datasets/henryhan117/sp-500-historical-data 
This is saved in the dataset folder as SPX_clean.csv, after cleaning. Additionally, synthetic datasets can be generated using the file named synthetic_dataset_generator.py inside dataset/synthetic folder by specifying the desired file name. 

### synthetic_dataset_generator.py
This code generates a synthetic dataset for ridge regression with a controlled condition number of the Hessian. The function generate_X_with_condition takes as input the feature dimension d, number of samples n, ridge penalty lam, and a desired condition number Q_desired. It constructs a design matrix X for a target condition number 1/n * (X^T X). The matrix is then scaled to lie within [-4, 4], but this is optional and can be removed. Additionally, a noise is introduced to mimic real-world data, adjustable under noise_std.

## Optimizer Implementations
All of the optimizers are implemented in the file names starting with the name of the optimizer. 
- sgd – Stochastic Gradient Descent (SGD) for ridge regression with noise analysis and convergence logging.
- svrg – Stochastic Variance Reduced Gradient (SVRG), following Johnson & Zhang (2013).
- asg – Accelerated Stochastic Gradient (ASG), inspired by Assran & Rabbat (2020). Includes spectral radius and Pareto analysis
There are two versions for each optimizer, one is designed for the real-world dataset and the other for synthetic, for the purposes of experimentation. However, they can be merged later.

Each files consist of the following overall flow:

The code implements and compares the algorithm with the closed-form ridge regression solution. The workflow begins by loading a synthetic dataset (features $X \in \mathbb{R}^{n \times d}$, target $y \in \mathbb{R}^n$) and splitting it into training and test sets. The ridge regression objective being minimized is

$$
P(w) \;=\; \frac{1}{n}\|Xw - y\|_2^2 \;+\; \lambda \|w\|_2^2,
$$

where $\lambda$ is the regularization parameter. The closed-form solution is computed as

$$
w_{\text{closed}} \;=\; (X^\top X + \lambda n I)^{-1} X^\top y,
$$

The updates are different for each algorithm:

### SGD 

$$
w_{k+1} \;=\; w_k - \eta \Big( 2 x_i^\top(x_i w_k - y_i) + 2\lambda w_k \Big),
$$

where $\eta$ is the learning rate. During training, the code logs the loss and the distance $\|w - w_{\star}\|$ from the closed-form optimum $w_{\star}$. It also computes Hessian eigenvalues to estimate the smoothness constant $L$, strong convexity $\mu$, condition number $Q = L / \mu$, and ideal step size $\alpha = 1/L$. Finally, the code visualizes convergence by plotting the loss against iterations (compared to the closed-form loss) and the norm gap $\|w - w_\star\|$. It also estimates the computational cost (FLOPs) of both the closed-form and SGD approaches to highlight the trade-off between accuracy and efficiency.

### SVRG 

SVRG proceeds in epochs with a snapshot $w_{\tilde{}}$ and its full gradient $\nabla P(w_{\tilde{}})=\tfrac{2}{n}X^\top(Xw_{\tilde{}}-y)+2\lambda w_{\tilde{}}$. Each inner step samples $(x_i,y_i)$ and applies the variance-reduced update

$$
w \leftarrow w-\eta\big(\underbrace{2x_i^\top(x_i w-y_i)+2\lambda w}_{\nabla_i P(w)}-\underbrace{(2x_i^\top(x_i w_{\tilde{}}-y_i)+2\lambda w_{\tilde{}})}_{\nabla_i P(w_{\tilde{}})}+\nabla P(w_{\tilde{}})\big),
$$

then the epoch output $w_{\tilde{}}^{\text{new}}$ is the average of inner iterates. The script logs the training loss $P(w)$ and the distance $\|w-w_{\text{closed}}\|$ each epoch, and performs spectral analysis of the ridge Hessian $H=\tfrac{2}{n}X^\top X+2\lambda I$ to report smoothness $L=\lambda_{\max}(H)$, strong convexity $\mu=\lambda_{\min}(H)$, condition number $Q=L/\mu$, and the ideal step $\alpha=1/L$. It also computes a convergence factor $\phi$ and prints the bound $\phi^{k}\big(P(w_0)-P(w_*)\big)$ over epochs $k$. Finally, the code plots the loss trajectory (with a horizontal line at the closed-form loss) and $\|w-w_*\|$, saves the figure, and reports FLOPs.

### ASG

The ASG step uses a lookahead point $u_k=w_k+\beta(w_k-w_{k-1})$ and applies a gradient at $u_k$:

$$
w_{k+1}=u_k-\alpha\Big(\tfrac{2}{|B|}X_B^\top(X_B u_k-y_B)+2\lambda u_k\Big)\.
$$

Although the update function is also written for the stochastic version, only the minibatch is used as it aligns with the theory, with batch size B.

In summary, the script 
(1) runs ASG-minibatch, saving loss and distance curves and comparing to the closed-form loss; 
(2) computes FLOPs for closed-form vs. ASG-minibatch to illustrate efficiency; and 
(3) generates $(\alpha,\beta)$ combinations to generate a Pareto front, keeping only stable pairs with $R_\lambda<1$, and visualizes performance via color maps of $\log(1+\|w-w_{\!*}\|)$ and $R_\lambda$.



Note: used ChatGPT for bettering the clarity.
