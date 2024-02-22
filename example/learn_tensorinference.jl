using TensorInference

model = read_model_file(pkgdir(TensorInference, "examples", "asia-network", "model.uai"))

tn = TensorNetworkModel(model)
