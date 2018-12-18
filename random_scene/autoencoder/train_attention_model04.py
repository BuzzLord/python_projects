from __future__ import print_function
from model04 import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    num_epochs = 2
    common_args = ["--model-path", "attention", "--epochs", str(num_epochs), "--seed", "1234"]
    logging_args = ["--log-file", "train_atten.log"]
    main(common_args + ["--epoch-start", str(1+0*num_epochs), "--attention", "e1", "4"] + logging_args)  # 1-2
    main(common_args + ["--epoch-start", str(1+1*num_epochs), "--attention", "e2", "2"])  # 3,4
    main(common_args + ["--epoch-start", str(1+2*num_epochs), "--attention", "e3", "1"])  # 5,6
    main(common_args + ["--epoch-start", str(1+3*num_epochs), "--attention", "e4", "1"])  # 7,8
    main(common_args + ["--epoch-start", str(1+4*num_epochs), "--attention", "e5", "1"])  # 9,10
    main(common_args + ["--epoch-start", str(1+5*num_epochs), "--attention", "d1", "1"])  # 11,12
    main(common_args + ["--epoch-start", str(1+6*num_epochs), "--attention", "d2", "1"])  # 13,14
    main(common_args + ["--epoch-start", str(1+7*num_epochs), "--attention", "d3", "1"])  # 15,16
    main(common_args + ["--epoch-start", str(1+8*num_epochs), "--attention", "d4", "2"])  # 17,18
    main(common_args + ["--epoch-start", str(1+9*num_epochs), "--attention", "d5", "4"])  # 19,20
