python3 main.py --save-model --epochs 1
python3 to_onnx.py 
python3 substitute_onnx.py
python3 print_onnx.py mnist_my_selu.onnx