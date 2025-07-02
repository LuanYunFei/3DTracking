import csv
import os.path

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np

plt.switch_backend('agg')


class CsvLogger:
    def __init__(self, filepath='./', filename='results.csv', data=None):
        self.log_path = filepath
        self.log_name = filename
        self.csv_path = os.path.join(self.log_path, self.log_name)
        self.fieldsnames = ['epoch', 'val_loss','train_loss','kl_loss','order_loss','rank_loss']

        with open(self.csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writeheader()

        self.data = {}
        for field in self.fieldsnames:
            self.data[field] = []
        if data is not None:
            for d in data:
                d_num = {}
                for key in d:
                    d_num[key] = float(d[key]) if key != 'epoch' else int(d[key])
                self.write(d_num)

    def write(self, data):
        for k in self.data:
            self.data[k].append(data[k])
        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)

    def save_params(self, args, params):
        with open(os.path.join(self.log_path, 'params.txt'), 'w') as f:
            f.write('{}\n'.format(' '.join(args)))
            f.write('{}\n'.format(params))

    def write_text(self, text, print_t=True):
        with open(os.path.join(self.log_path, 'params.txt'), 'a') as f:
            f.write('{}\n'.format(text))
        if print_t:
            print(text)