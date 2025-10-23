import utils as utils
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file

    replaced_file = utils.replace_pod_ip_with_generalpodid(input_file)
    print(replaced_file)
