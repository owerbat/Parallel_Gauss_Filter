import numpy as np


def calculate_average():
    with open('../Results/txt/times.txt', 'r') as source_file:
        source_file.readline()

        lines = np.array([np.array(line.split(), dtype=float) 
                          for line in source_file.readlines()]).T
        averages = [np.mean(line) for line in lines]

        with open('../Results/txt/average_times.txt', 'w') as result_file:
            for method, value in zip(['Simple', 'OpenMP', 'TBB'], averages):
                result_file.write(f'{method}\t{value}\n')


if __name__ == '__main__':
    calculate_average()
