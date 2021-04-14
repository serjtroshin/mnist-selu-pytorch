import onnx

onnx_model = onnx.load('mnist_cnn.onnx')

endpoint_names = ['Selu', 'mySelu']

for i in range(len(onnx_model.graph.node)):
    if hasattr(onnx_model.graph.node[i], "op_type"):
        if onnx_model.graph.node[i].op_type == endpoint_names[0]:
            onnx_model.graph.node[i].op_type = endpoint_names[1]

onnx.save(onnx_model, 'mnist_my_selu.onnx')