using ComputationalResources
using MLJ
using DelimitedFiles
@load SVC pkg = LIBSVM
#@load SVMClassifier pkg = ScikitLearn

# using MLDatasets
#
# train_X, train_y = MNIST.traindata(Float64)
# test_X, test_y = MNIST.testdata(Float64)
#
# tr_X = Array{Float64,2}(
#     undef,
#     size(train_X, 3),
#     size(train_X, 1) * size(train_X, 2),
# )
# for i = 1:size(train_X, 3)
#     tr_X[i, :] = collect(Iterators.flatten(train_X[:, :, i]))
# end
#
# ts_X =
#     Array{Float64,2}(undef, size(test_X, 3), size(test_X, 1) * size(test_X, 2))
# for i = 1:size(test_X, 3)
#     ts_X[i, :] = collect(Iterators.flatten(test_X[:, :, i]))
# end

tr = readdlm("ml/data/optdigitstra.csv", ',')
tr_X = tr[:,1:64]
tr_y = tr[:,65]

ts = readdlm("ml/data/optdigitstes.csv", ',')
ts_X = ts[:,1:64]
ts_y = ts[:,65]

model = SVC(gamma = 0.001)
#model = SVMClassifier(gamma = 0.001)

#r = range(model, :gamma, lower = 0.00075, upper = 0.00125)
#st_svm = TunedModel(model = model, range = r, tuning = Grid(resolution = 80), acceleration=CPU1(), acceleration_resampling=CPU1())

train_pred = table(tr_X)
train_out = categorical(tr_y)

test_pred = table(ts_X)
test_out = categorical(ts_y)
mach = machine(model, train_pred, train_out)
#mach = machine(st_svm, train_pred, train_out)

fit!(mach)
#fitted_params(mach).best_model

yp = predict(mach, test_pred)
mcr = misclassification_rate(yp, test_out)
