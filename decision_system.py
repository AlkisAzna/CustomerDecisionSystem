import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt


class DecisionSystem:
    def __init__(self, file):
        self.df = self._load(file)
        self.order_data = self.df.loc["Data Order", :]
        self.df = self.df.iloc[:-1]
        self.total_chars = self.get_total_characteristics()
        self.total_customers = self.get_total_customers()
        self.total_options = self.get_options()
        self.max_spaces = 5
        self.s_count = 1
        self.distance_index = self.find_distance_indexes()
        self.weights_per_option = self.calculate_weights()
        self.temp_weights = self.calculate_weights()
        self.weights_per_ranking = self.calculate_ranking_weights()
        self.options_usage = self.create_usage_list()
        self.imported_data = {}
        self.lindo_F = 0.0

    @staticmethod
    def _load(data):
        if isinstance(data, (str, Path)):
            return pd.read_excel(data, header=1).iloc[:-1].set_index("Car")
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise TypeError(f"Unsupported init type: {type(data)}")

    # Initital methods processing data
    def get_value(self, row, col):
        return self.df.loc[row, col]

    def get_total_characteristics(self):
        return self.df.columns.get_loc("Ranking1")  # Format of excel is specific

    def get_total_customers(self):
        # Calculated by TotalNumberColumns - (totalChars + 1) / 2
        # totalChars + 1 = Because index starts from zero
        # /2 =Because every customer sets 2 opinions for design and ranking
        return int((len(self.df.columns) - (self.total_chars - 1)) / 2)

    def get_options(self):
        return len(self.df.index)

    def find_distance_indexes(self):
        index_dictionary = {}

        for idx, val in enumerate(self.df.columns):
            if "Ranking" in val:
                continue
            elif "Design" in val:
                if "Design" in index_dictionary:
                    continue
                index_dictionary["Design"] = self.calculate_point_distances(val)
            else:
                index_dictionary[val] = self.calculate_point_distances(val)
                if "Acceleration" in val:
                    index_dictionary[val] = index_dictionary[val] - 1

        return index_dictionary

    def create_usage_list(self):
        usage_dict = {}
        for x in range(self.total_customers):
            usage_dict["DM" + str(x + 1)] = {}
            for idx, y in enumerate(self.df.index):
                usage_dict["DM" + str(x + 1)][y] = 0.0

        return usage_dict

    # Calculation methods
    def calculate_point_distances(self, col_name):
        max_value = max(self.df[col_name])
        min_value = min(self.df[col_name])
        equal_spaces = 1
        range_values = round(max_value - min_value)

        if range_values < (self.get_options() - 2):
            if range_values >= self.max_spaces:
                equal_spaces = self.max_spaces
            else:
                equal_spaces = range_values
        else:
            i = self.max_spaces
            while i > 1:
                if (range_values % i) == 0:
                    equal_spaces = i
                    if i == 3:
                        break
                i = i - 1

        return equal_spaces

    # Calculate the function of each customer of each option depending on characteristics
    def calculate_weights(self):
        option_weights = {}

        for x in range(self.total_customers):
            option_weights["DM" + str(x + 1)] = {}

            for index, row in enumerate(self.df.index):
                option_weights["DM" + str(x + 1)][row] = ""
                for idx, col in enumerate(self.df.columns):
                    if "Design1" in col:
                        col = "Design" + str(x + 1)
                    elif "Design" in col:
                        continue

                    if "Ranking" in col:
                        continue

                    if self.order_data[col] == "INCR":
                        max_value = max(self.df[col])
                        min_value = min(self.df[col])
                    else:
                        max_value = min(self.df[col])
                        min_value = max(self.df[col])
                    range_values = abs(max_value - min_value)

                    if "Design" in col:
                        spacing = range_values / self.distance_index["Design"]
                    else:
                        spacing = range_values / self.distance_index[col]

                    if self.order_data[col] == "DECR":
                        spacing = -spacing

                    current_value = self.get_value(row, col)
                    temp_value = min_value
                    count = 1

                    if current_value == min_value:
                        if "Design" in col:
                            option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                row) + str(0.0)
                        continue
                    else:
                        while True:
                            if ((spacing + temp_value) >= current_value and self.order_data[col] == "INCR") or \
                                    ((spacing + temp_value) <= current_value and self.order_data[col] == "DECR"):
                                if "Design" in col:
                                    option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                        row) + str(
                                        round(abs((current_value - temp_value) / spacing), 2)) + " * w" + str(
                                        idx + 1) + str(count)
                                else:
                                    option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                        row) + str(
                                        round(abs((current_value - temp_value) / spacing), 2)) + " * w" + str(
                                        idx + 1) + str(count) + " + "
                                break
                            else:
                                option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                    row) + "w" + str(
                                    idx + 1) + str(count) + " + "
                                count += 1
                                temp_value += spacing

        return option_weights

    # Calculate the function of each customer of each option depending on ranking
    def calculate_ranking_weights(self):
        ranking_weights = {}
        # Make type integer of ranking so that it was be sorted
        for idx, col in enumerate(self.df.columns):
            if "Ranking" in col:
                self.df[col] = self.df[col].astype(int)

        for x in range(self.total_customers):
            ranking_weights["DM" + str(x + 1)] = {}
            temp_df = self.df.sort_values(by=("Ranking" + str(x + 1)), ascending=True)
            current_col_index = temp_df.columns.get_loc("Ranking" + str(x + 1))
            for index, row in enumerate(temp_df.index):
                if temp_df.iloc[index, current_col_index] == temp_df.iloc[index + 1, current_col_index]:
                    ranking_weights["DM" + str(x + 1)]["D(" + row + "-" + temp_df.index[index + 1]] = \
                        self.weights_per_option["DM" + str(x + 1)][row] + " - (" + \
                        self.weights_per_option["DM" + str(x + 1)][temp_df.index[index + 1]] + ") - s" + str(
                            self.s_count) + " + s" + str(self.s_count + 1) + " + s" + str(
                            self.s_count + 2) + " - s" + str(
                            self.s_count + 3) + " = 0.0"
                else:  # Greater only because it is sorted
                    ranking_weights["DM" + str(x + 1)]["D(" + row + "-" + temp_df.index[index + 1]] = \
                        self.weights_per_option["DM" + str(x + 1)][
                            row] + " - (" + self.weights_per_option["DM" + str(x + 1)][
                            temp_df.index[index + 1]] + ") - s" + str(self.s_count) + " + s" + str(
                            self.s_count + 1) + " + s" + str(self.s_count + 2) + " - s" + str(
                            self.s_count + 3) + " >= 0.05"
                self.s_count += 2
                if index == self.get_options() - 2:
                    break
            self.s_count += 2

        return ranking_weights

    def process_import_data(self):
        temp_weights_option = self.weights_per_option
        if "F" in self.imported_data:
            self.lindo_F = self.imported_data["F"]
        for x in range(self.total_customers):
            for index, row in enumerate(self.df.index):
                for idx, val in enumerate(self.imported_data["Weights"]):
                    for i, k_val in enumerate(self.imported_data["Weights"][val]):
                        if k_val in temp_weights_option["DM" + str(x + 1)][row]:
                            temp_value = temp_weights_option["DM" + str(x + 1)][row].replace(k_val, self.imported_data[
                                "Weights"][val][k_val])
                            temp_weights_option["DM" + str(x + 1)][row] = temp_value
                self.options_usage["DM" + str(x + 1)][row] = eval(temp_weights_option["DM" + str(x + 1)][row])
        self.imported_data = {}
        self.weights_per_option = self.temp_weights
        return

    # Import-Export data
    def import_lindo_settings(self, file):
        with open(file) as f:
            data = json.load(f)
        self.imported_data = data
        self.process_import_data()
        return

    def export_lindo_data(self):
        temp_dict = self.weights_per_ranking
        temp_dict["F"] = "min{Sum(si) from i=1 to " + str(self.s_count - 1) + "}"
        temp_dict[
            "Conditions"] = "Sum(wij) = 1, s(i), w(ij) >= 0 where i = " + str(
            self.get_total_characteristics()) + " and j = distance for each characteristic"
        with open('Initial_Lindo_Settings.json', 'w') as outfile:
            json.dump(temp_dict, outfile)
        return

    def export_optimizer_data(self, e_val):
        temp_dict = self.weights_per_ranking
        temp_dict["F"] = {}
        for x in range(self.total_chars):
            for idx, y in enumerate(self.distance_index):
                for index in range(self.distance_index[y]):
                    if index == 0:
                        temp_dict["F"][str(x + 1)] = "w" + str(idx + 1) + str(index + 1)
                    else:
                        temp_dict["F"][str(x + 1)] = temp_dict["F"][str(x + 1)] + "w" + str(idx + 1) + str(index + 1)
            temp_dict["F"][str(x + 1)] = "max{" + temp_dict["F"][str(x + 1)] + "}"
            temp_dict["Condition1"] = "Sum(wij) = 1, s(i), w(ij) >= 0 where i = " + str(
                self.get_total_characteristics()) + " and j = distance for each characteristic"
            lindo_val = (1.0 + e_val) * float(self.lindo_F)
            temp_dict["Condition2"] = "Sum(si)<=" + str(lindo_val) + " from i=1 to " + str(
                self.s_count - 1)
            with open('Optimization_Lindo_Settings.json', 'w') as outfile:
                json.dump(temp_dict, outfile)
        return

    def export_in_excel(self, filename):
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df = pd.DataFrame(self.options_usage.values(),
                              index=self.options_usage.keys())
            df.to_excel(writer, sheet_name='Usage List')
        print('Excel file with usage list created Successfully')
        return

    def create_figures(self, filename, set_title):
        data = pd.DataFrame(self.options_usage.values(), index=self.options_usage.keys())
        data.plot(kind='bar', figsize=(14, 9), xlabel='Customers', ylabel='Usage Value',
                  title=set_title)
        plt.legend(frameon=False, loc='upper center', ncol=7)
        plt.savefig(filename)
        plt.show()
