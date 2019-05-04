bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package &> log.txt
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
#pip uninstall tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-1.13.1-cp27-cp27mu-linux_x86_64.whl


