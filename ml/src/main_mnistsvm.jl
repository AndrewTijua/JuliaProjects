using ComputationalResources
using MLJ
using DelimitedFiles
@load SVC pkg = LIBSVM
#@load SVMClassifier pkg = ScikitLearn

using MLDatasets

include("libs/long_matrix_utils.jl")

train_X, tr_y = MNIST.traindata(Float64)
test_X, ts_y = MNIST.testdata(Float64)

ts_X = matrix_array_to_long_list(test_X)


#lm = tr_X
# open("ml/data/opdig_lrg_tr.csv", "w") do io
#     writedlm(io, lm[1:100, :], ',')
# end

# tr = readdlm("ml/data/optdigitstra.csv", ',')
# tr_X = tr[:, 1:64]
# tr_y = tr[:, 65]
#
# ts = readdlm("ml/data/optdigitstes.csv", ',')
# ts_X = ts[:, 1:64]
# ts_y = ts[:, 65]

tr_sz = 50000
ts_sz = 10000


train_X = train_X[:, :, 1:tr_sz]
ts_X = ts_X[1:ts_sz, :]

tr_y = tr_y[1:tr_sz]
ts_y = ts_y[1:ts_sz]

tr_X = matrix_array_to_long_list(train_X)

tr_X_22 = long_matrix_subaverage(tr_X, [28, 28], [2, 2])
ts_X_22 = long_matrix_subaverage(ts_X, [28, 28], [2, 2])
tr_X_33 = long_matrix_subaverage(tr_X, [28, 28], [3, 3])
ts_X_33 = long_matrix_subaverage(ts_X, [28, 28], [3, 3])
tr_X_44 = long_matrix_subaverage(tr_X, [28, 28], [4, 4])
ts_X_44 = long_matrix_subaverage(ts_X, [28, 28], [4, 4])

tr_X = hcat(tr_X_44, tr_X_33, tr_X_22)
ts_X = hcat(ts_X_44, ts_X_33, ts_X_22)



num_features = size(tr_X, 2)

#model = SVC(gamma = 0.001)
model = SVC(gamma = 41.3 * 1 / num_features)
#model = SVMClassifier(gamma = -1.)
#model = SVMClassifier(gamma = 0.001)


#r = range(model, :gamma, lower = 38 * 1/num_features, upper = 43 * 1/num_features)
#st_svm = TunedModel(model = model, range = r, tuning = Grid(resolution = 10), acceleration=CPUThreads(), acceleration_resampling=CPU1())

train_pred = table(tr_X)
train_out = categorical(tr_y)

test_pred = table(ts_X)
test_out = categorical(ts_y)
mach = machine(model, train_pred, train_out)
# mach = machine(st_svm, train_pred, train_out)

fit!(mach)
# fi_p = fitted_params(mach).best_model

yp = predict(mach, test_pred)
mcr = misclassification_rate(yp, test_out)
conf = confusion_matrix(yp, test_out)

pcm = conf[:, :]
dvs = sum(pcm, dims = 1)
props = pcm ./ dvs
