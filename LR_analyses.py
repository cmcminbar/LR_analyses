# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate


all_vars = ['LR_Group', 'age', 'gender', 'SP_U', 'RA_AMP_U', 'LA_AMP_U', 'RA_STD_U',
       'LA_STD_U', 'SYM_U', 'R_JERK_U', 'L_JERK_U', 'ASA_U', 'ASYM_IND_U',
       'TRA_U', 'T_AMP_U', 'CAD_U', 'STR_T_U', 'STR_CV_U', 'STEP_REG_U',
       'STEP_SYM_U', 'JERK_T_U', 'SP_DT', 'RA_AMP_DT', 'LA_AMP_DT',
       'RA_STD_DT', 'LA_STD_DT', 'SYM_DT', 'R_JERK_DT', 'L_JERK_DT', 'ASA_DT',
       'ASYM_IND_DT', 'TRA_DT', 'T_AMP_DT', 'CAD_DT', 'STR_T_DT', 'STR_CV_DT',
       'STEP_REG_DT', 'STEP_SYM_DT', 'JERK_T_DT', 'SW_VEL_OP', 'SW_PATH_OP',
       'SW_FREQ_OP', 'SW_JERK_OP', 'SW_VEL_CL', 'SW_PATH_CL', 'SW_FREQ_CL',
       'SW_JERK_CL', 'TUG1_DUR', 'TUG1_STEP_NUM', 'TUG1_TURNS_DUR', 'TUG1_STEP_REG', 'TUG1_STEP_ASYM', 'TUG2_DUR',
       'TUG2_STEP_NUM', 'TUG2_TURNS_DUR', 'TUG2_STEP_REG',
       'TUG2_STEP_ASYM']

# read in CSVs
pd.options.display.max_rows = 1000
data_dir = 'N:\Gait-Neurodynamics by Names\Inbar\Beat ML\BeatReports\PPMI_format'
save_dir = 'processes data'
df_gait: pd.DataFrame = pd.read_csv(
    os.path.join(data_dir, save_dir, 'BEATandPPMIReport_29-Jun-2021.csv'))

df_gait = df_gait[all_vars]
is_plt = True
def plot_vars(X, Y, dataset, title=None, x_label='X', y_label=None):
    """
    This function is for plotting selected variables.

    Parameters
    ----------
    dataset - A pandas dataframe with variables
    X - x axis variable
    Y - y axis variable
    kind - type of plot (cat- categorical and scat - scatter)
    title - title of the plot
    x_label - x label of the plot
    y_label - y label of the plot

    Returns
    ---------
    Returns None.
    Plots a categorical bar chart of the variables with the data scattered on the Y axis.

    """
    plt.figure(figsize=(6, 4))
    sns.barplot(x=X, y=Y, data=dataset, alpha=0.3)
    sns.stripplot(x=X, y=Y, data=dataset, alpha=0.5)
    plt.xlabel(str(x_label), fontsize=12)
    plt.ylabel(str(y_label), fontsize=12)
    plt.title(str(title), fontsize=15)
    plt.show()


# let's plot some data: age, single speed vel, dual task vel, stride time var single, stride time var dual)
if is_plt:
    dv = ['age', 'SP_U', 'SP_DT', 'CAD_DT', 'STR_T_U', 'STR_T_DT', 'STR_CV_U', 'STR_CV_DT', 'SW_VEL_CL', 'SW_FREQ_OP',
        'TUG1_DUR','TUG2_DUR','TUG1_TURNS_DUR','TUG2_TURNS_DUR']
    for i in range(len(dv)):
        plot_vars('LR_Group', dv[i], df_gait, title=dv[i], x_label='Groups')

# drop rows with nans in target
df_sparse = df_gait.dropna(subset=['LR_Group'])
# this cell should have code for creating different dataframes depending on how we want to handle missing observations
df_sparse = df_sparse.dropna(subset=all_vars, thresh=0.6*len(all_vars))
df_sparse = df_sparse.dropna(axis=1, thresh=0.9*len(df_sparse))
df_sparse.to_csv(os.path.join(data_dir, save_dir, 'df_sparse_LRGroup.csv'))
df_sparse.to_excel(os.path.join(data_dir, save_dir, 'df_sparse_LRGroup.xlsx'))
# Descriptive statistics split by group
df_sparse.groupby('LR_Group').describe()

# define dataset - looking only on low vs high LR
y = df_sparse['LR_Group'][(df_sparse['LR_Group']==0) | (df_sparse['LR_Group']==2)]
X = df_sparse.drop(columns=['LR_Group', 'age', 'gender'])
X = X[(df_sparse['LR_Group']==0) | (df_sparse['LR_Group']==2)]
y = LabelEncoder().fit_transform(y)

# perform cross-validation to find best lambda
nFolds = 5
lambdas = np.arange(0.1, 15.01, 0.1)
C_vals = 1 / lambdas
shrinkage_factor = np.linspace(0, 1, len(lambdas))  # for finding optimal regularizing parameter in LDA

# store dictionaries of cross-validation results (train, test, estimator) in a list
cv_results = []
cv_results_lda = []
train_scores = np.zeros((nFolds, len(lambdas)))
train_scores_lda = np.zeros((nFolds, len(lambdas)))
test_scores = np.zeros((nFolds, len(lambdas)))
test_scores_lda = np.zeros((nFolds, len(lambdas)))
betas_lr = np.empty((nFolds, len(lambdas)), dtype=object)

# define the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

for i in range(len(lambdas)):

    # define pipeline for logistic regression
    steps = list()
    steps.append(('imputer', imputer))
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=int(1e6), \
                                              C=C_vals[i], warm_start=True)))
    pipeline = Pipeline(steps=steps)

    # define pipeline for linear discriminant analysis
    steps_lda = list()
    steps_lda.append(('imputer', imputer))
    steps_lda.append(('scaler', StandardScaler()))
    # shrinkage factor for LDA takes on value between 0-1 and is a form of regularization:
    # 0 shrinkage amounts to using empirical covariance, 1 amounts to using diagonal matrix
    # of variances as estimate of covariance (setting shrinkage argument to 'auto' finds the
    # best analytic solution)
    steps_lda.append(('model', LDA(shrinkage=shrinkage_factor[i], solver='lsqr')))
    pipeline_lda = Pipeline(steps=steps_lda)

    # define the evaluation procedure
    cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=123)

    # evaluate the models using cross-validation
    cv_results.append(cross_validate(pipeline, X, y, scoring='balanced_accuracy', cv=cv, \
                                     n_jobs=1, return_estimator=True, return_train_score=True))
    cv_results_lda.append(cross_validate(pipeline_lda, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=1,
                                         return_estimator=True, return_train_score=True))

    train_scores[:, i] = cv_results[i]['train_score']
    train_scores_lda[:, i] = cv_results_lda[i]['train_score']

    test_scores[:, i] = cv_results[i]['test_score']
    test_scores_lda[:, i] = cv_results_lda[i]['test_score']

    for j in range(nFolds):
        betas_lr[j, i] = cv_results[i]['estimator'][j]['model'].coef_.flatten()  # flatten makes list of coef values 1d

if is_plt:
    plt.style.use('ggplot')

    # plot training and test scores for regularized LR
    f = plt.figure(figsize=(6, 4))
    plt.plot(lambdas, train_scores.mean(axis=0) * 100, color='blue', linewidth=3, label='Training')
    plt.plot(lambdas, test_scores.mean(axis=0) * 100, color='red', linewidth=3, label='Test')
    for i in range(train_scores.shape[0]):
        plt.plot(lambdas, train_scores[i, :] * 100, color='blue', linewidth=0.5, alpha=0.6)
        plt.plot(lambdas, test_scores[i, :] * 100, color='red',linewidth=0.5, alpha=0.6)
    plt.axvline(lambdas[np.argmax(test_scores.mean(axis=0))], c='k', linewidth=0.5, linestyle='--')
    plt.xlabel('$\lambda$')
    plt.ylabel('Classification accuracy (%)')
    plt.ylim([0, 102.5])
    plt.title('LASSO Logistic Regression')
    plt.legend(frameon=False)
    f.savefig(os.path.join(data_dir, 'logistic_cv.png'), dpi=300, bbox='tight')

    # plot training and test scores for regularized LDA
    f = plt.figure(figsize=(6, 4))
    plt.plot(shrinkage_factor, train_scores_lda.mean(axis=0) * 100, color='blue', linewidth=3, label='Training')
    plt.plot(shrinkage_factor, test_scores_lda.mean(axis=0) * 100, color='red', linewidth=3, label='Test')
    for i in range(train_scores.shape[0]):
        plt.plot(shrinkage_factor, train_scores_lda[i, :] * 100, color='blue', linewidth=0.5, alpha=0.6)
        plt.plot(shrinkage_factor, test_scores_lda[i, :] * 100, color='red', linewidth=0.5, alpha=0.6)
    plt.axvline(shrinkage_factor[np.argmax(test_scores_lda.mean(axis=0))], c='k', linewidth=0.5, linestyle='--')
    plt.xlabel('Shrinkage factor')
    plt.ylabel('Classification accuracy (%)')
    plt.ylim([0, 102.5])
    plt.title('Regularized Linear Discriminant')
    plt.legend(frameon=False)
    f.savefig(os.path.join(data_dir, 'Output\lda_cv.png'), dpi=300, bbox='tight')

# find optimal regularization factors and betas
idx = np.argmax(np.mean(test_scores, axis=0))
idx_lda = np.argmax(np.mean(test_scores_lda, axis=0))
print(f'The optimal $\lambda$ for logistic regression is: {lambdas[idx]:.2f}')
print(f'The highest mean test accuracy for logistic regression is: {test_scores[:, idx].mean()}')
print(f'The optimal shrinkage factor for LDA is: {shrinkage_factor[idx_lda]:.2f}')
print(f'The highest mean test accuracy for LDA is: {test_scores_lda[:, idx_lda].mean()}')

beta_lr_opt = np.stack(betas_lr[:, idx])
features = X.columns
if is_plt:
    # find betas using index for optimal lambda
    f = plt.figure(figsize=(20, 4))
    plt.bar(features, (np.count_nonzero(beta_lr_opt, axis=0) / 5) * 100)
    plt.xticks(rotation=90)
    plt.ylabel('% of times present in best-fit model')
    plt.title('Predictors in logistic regression model')
    f.savefig(os.path.join(data_dir, 'Output\predictors_hist.png'), dpi=300, bbox_inches='tight')

    # plot beta weights
    f = plt.figure(figsize=(20, 6))
    plt.boxplot(beta_lr_opt)
    plt.xticks(range(1, len(features) + 1), labels=features, rotation=90)
    plt.ylabel('Beta weights')
    f.savefig(os.path.join(data_dir, 'Output\llasso_weights.png'), dpi=300, bbox_inches='tight')


