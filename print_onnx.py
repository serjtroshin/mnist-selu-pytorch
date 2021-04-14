from argparse import ArgumentParser
import onnx

parser = ArgumentParser()
parser.add_argument("name", type=str)
args = parser.parse_args()

name = args.name
ir = onnx.load(name)
print(onnx.helper.printable_graph(ir.graph))