import argparse


def arg_helper() -> list:
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-m", "--model", default=None, help="path to Model File ")
    arg_parse.add_argument("-d", "--data", default=None, help="path to Data File ")
    arg_parse.add_argument("-o", "--output", default=None, help="path to Output File ")
    args = vars(arg_parse.parse_args())
    return args